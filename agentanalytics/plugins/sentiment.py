import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict

import numpy as np

from ..core.plugin import PluginResult, MetricRecord, Artifact, TraceAnnotation
from ..core.model import TraceBatch
from ..core.view import TraceView
from ..utils.textnorm import norm_numbers, shorten
from ..utils.clustering import cluster_embeddings, pick_representative_index


def _ensure_hf_cache(cache_root: str) -> None:
    """
    Make HF/Transformers caching persistent and consistent.
    """
    cache_root = os.path.expanduser(cache_root)
    hf_home = os.path.join(cache_root, "hf")
    os.makedirs(hf_home, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))


def _device_arg(device: str) -> int:
    """
    transformers pipeline uses device: -1 for CPU, >=0 for GPU index
    """
    d = (device or "cpu").lower()
    if d == "cpu":
        return -1
    if d.startswith("cuda"):
        # "cuda" or "cuda:0"
        if ":" in d:
            return int(d.split(":")[1])
        return 0
    return -1


def _score_from_pipeline_outputs(outputs: Any) -> Tuple[str, float]:
    """
    Normalize sentiment to:
      label in {"positive","neutral","negative"}
      score in [-1, 1] (signed)
    Handles:
      - binary POS/NEG (SST-2): pipeline returns [{'label':'POSITIVE','score':0.99}]
      - 3-class (TweetEval-like): when return_all_scores=True
    """
    # Case 1: list[dict] with top label only
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict) and "label" in outputs[0]:
        lab = str(outputs[0]["label"]).lower()
        sc = float(outputs[0].get("score", 0.0))
        if "pos" in lab:
            return "positive", float(np.clip(sc, 0, 1))
        if "neg" in lab:
            return "negative", float(np.clip(-sc, -1, 0))
        # sometimes "neutral" appears as top label
        if "neu" in lab:
            return "neutral", 0.0
        return "neutral", 0.0

    # Case 2: return_all_scores=True => list[list[dict]]
    # Example: [[{'label':'LABEL_0','score':...}, ...]]
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], list):
        scores = outputs[0]
        # map labels by name if possible
        pos = 0.0
        neg = 0.0
        neu = 0.0
        for item in scores:
            lab = str(item.get("label", "")).lower()
            sc = float(item.get("score", 0.0))
            if "pos" in lab:
                pos = max(pos, sc)
            elif "neg" in lab:
                neg = max(neg, sc)
            elif "neu" in lab:
                neu = max(neu, sc)

        # If labels are numeric (LABEL_0/1/2), we can’t reliably infer mapping.
        # In that case, fall back to "top label only" behavior by picking max.
        if pos == 0.0 and neg == 0.0 and neu == 0.0:
            top = max(scores, key=lambda x: float(x.get("score", 0.0)))
            return _score_from_pipeline_outputs([top])

        signed = float(np.clip(pos - neg, -1.0, 1.0))
        if signed >= 0.25:
            return "positive", signed
        if signed <= -0.25:
            return "negative", signed
        return "neutral", signed

    return "neutral", 0.0


@dataclass
class SentimentAnalyzer:
    name: str = "sentiment_analyzer"
    version: str = "0.1.0"

    def analyze(self, batch: TraceBatch, ctx, config: Dict[str, Any]) -> PluginResult:
        # ---- config ----
        source = str(config.get("source", "query_text"))  # "query_text" | "final_answer"
        model_id = str(config.get("model_id", "distilbert/distilbert-base-uncased-finetuned-sst-2-english"))
        cache_root = str(config.get("cache_root", "~/.cache/agentanalytics"))
        device = str(config.get("device", "cpu"))  # "cpu" | "cuda" | "cuda:0"
        batch_size = int(config.get("batch_size", 32))
        top_k_negative = int(config.get("top_k_negative", 15))
        include_negative_clustering = bool(config.get("include_negative_clustering", True))
        min_cluster_size = int(config.get("min_cluster_size", 10))
        min_samples = int(config.get("min_samples", 3))

        _ensure_hf_cache(cache_root)

        # ---- load pipeline ----
        try:
            from transformers import pipeline
        except Exception as e:
            raise RuntimeError(
                "transformers/torch not installed. Install with: pip install -e '.[sentiment]'"
            ) from e

        clf = pipeline(
            task="sentiment-analysis",
            model=model_id,
            device=_device_arg(device),
        )  # pipelines are the recommended inference API. :contentReference[oaicite:2]{index=2}

        views: List[TraceView] = [ctx.view_factory.make(t) for t in batch.traces]

        texts: List[str] = []
        trace_ids: List[str] = []
        timestamps: List[int] = []

        for t, v in zip(batch.traces, views):
            text = (v.final_answer() if source == "final_answer" else v.query_text()) or ""
            text = text.strip()
            if not text:
                continue
            texts.append(text)
            trace_ids.append(v.trace_id)
            timestamps.append(int(t.start_time_ms))

        total = len(texts)
        if total == 0:
            summary_md = f"ℹ️ No texts available for sentiment scoring (source: `{source}`)."
            return PluginResult(
                metrics=[MetricRecord(name="sentiment.summary", data={"total": 0})],
                artifacts=[],
                summary_md=summary_md,
            )

        # ---- inference (batched) ----
        labels: List[str] = []
        scores: List[float] = []

        for i in range(0, total, batch_size):
            chunk = texts[i : i + batch_size]
            out = clf(chunk)
            # pipeline returns list of dicts for batch
            for item in out:
                lab, sc = _score_from_pipeline_outputs([item])
                labels.append(lab)
                scores.append(sc)

        label_counts = Counter(labels)
        neg_idx = [i for i, lab in enumerate(labels) if lab == "negative"]
        pos_idx = [i for i, lab in enumerate(labels) if lab == "positive"]

        neg_rate = (len(neg_idx) / total) if total else 0.0
        pos_rate = (len(pos_idx) / total) if total else 0.0

        # Most negative examples by signed score (more negative -> smaller)
        neg_examples = sorted(
            [(scores[i], trace_ids[i], texts[i]) for i in neg_idx],
            key=lambda x: x[0],
        )[:top_k_negative]

        # Optional clustering of negative themes
        neg_clusters = []
        if include_negative_clustering and len(neg_examples) >= max(8, min_cluster_size):
            neg_texts = [norm_numbers(x[2]) for x in neg_examples]
            emb = ctx.resources.embedder.embed(neg_texts)  # reuse your existing embedder
            cr = cluster_embeddings(emb, min_cluster_size=min_cluster_size, min_samples=min_samples, seed=ctx.limits.random_seed)
            byc = defaultdict(list)
            for j, lab in enumerate(cr.labels.tolist()):
                if int(lab) == -1:
                    continue
                byc[int(lab)].append(j)

            for cid, idxs in byc.items():
                rep_j = pick_representative_index(emb, idxs)
                neg_clusters.append({
                    "cluster_id": cid,
                    "count": len(idxs),
                    "rep_text": neg_examples[rep_j][2],
                    "example_trace_ids": [neg_examples[k][1] for k in idxs[:5]],
                })
            neg_clusters.sort(key=lambda c: c["count"], reverse=True)

        # Annotations
        annotations: List[TraceAnnotation] = []
        per_trace = []
        for i in range(total):
            annotations.append(
                TraceAnnotation(
                    trace_id=trace_ids[i],
                    annotations={
                        "sentiment.label": labels[i],
                        "sentiment.score": float(scores[i]),
                        "sentiment.source": source,
                        "sentiment.model_id": model_id,
                    },
                )
            )
            per_trace.append({
                "trace_id": trace_ids[i],
                "timestamp_ms": timestamps[i],
                "label": labels[i],
                "score": float(scores[i]),
                "text": texts[i],
            })

        # Summary MD (generic renderer)
        summary_md = (
            f"**Sentiment distribution** (model: `{model_id}`, source: `{source}`)\n\n"
            f"- Positive: **{pos_rate*100:.1f}%** ({label_counts['positive']}/{total})\n"
            f"- Neutral: **{(label_counts['neutral']/total)*100:.1f}%** ({label_counts['neutral']}/{total})\n"
            f"- Negative: **{neg_rate*100:.1f}%** ({label_counts['negative']}/{total})\n\n"
        )

        if label_counts["negative"] == 0:
            summary_md += "✅ **No negative sentiment detected** in this window.\n"
        else:
            summary_md += "**Most negative examples:**\n"
            for i, (s, tid, txt) in enumerate(neg_examples[:3], 1):
                summary_md += f"{i}. `{tid}` — {shorten(txt, 120)}\n"
            if neg_clusters:
                summary_md += "\n**Top negative themes:**\n"
                for c in neg_clusters[:2]:
                    summary_md += f"- {shorten(c['rep_text'], 120)} (**{c['count']}**) \n"

        summary = {
            "total": total,
            "positive": int(label_counts["positive"]),
            "neutral": int(label_counts["neutral"]),
            "negative": int(label_counts["negative"]),
            "neg_rate": float(neg_rate),
            "pos_rate": float(pos_rate),
            "model_id": model_id,
            "source": source,
        }

        # Artifacts
        os.makedirs(ctx.output_dir, exist_ok=True)
        json_path = os.path.join(ctx.output_dir, "sentiment.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "label_counts": dict(label_counts),
                "negative_clusters": neg_clusters,
                "most_negative_examples": [
                    {"score": s, "trace_id": tid, "text": txt} for (s, tid, txt) in neg_examples
                ],
                "per_trace": per_trace,
            }, f, indent=2)

        md_path = os.path.join(ctx.output_dir, "sentiment_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Sentiment & (Un-)Satisfaction Report\n\n")
            f.write(summary_md + "\n")

        ann_path = os.path.join(ctx.output_dir, "sentiment_trace_annotations.json")
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump([{"trace_id": a.trace_id, "annotations": a.annotations} for a in annotations], f, indent=2)

        return PluginResult(
            metrics=[
                MetricRecord(name="sentiment.summary", data=summary),
                MetricRecord(name="sentiment.label_counts", data=dict(label_counts)),
            ],
            artifacts=[
                Artifact(kind="markdown", path=md_path, description="Sentiment report"),
                Artifact(kind="json", path=json_path, description="Sentiment JSON"),
                Artifact(kind="json", path=ann_path, description="Per-trace sentiment annotations"),
            ],
            annotations=annotations,
            summary_md=summary_md,
            summary=summary,
        )