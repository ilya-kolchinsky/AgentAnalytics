import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from ..core.model import TraceBatch
from ..core.plugin import Artifact, MetricRecord, PluginResult, TraceAnnotation
from ..core.view import TraceView


def _normalize_query(q: str) -> str:
    q = q.strip().lower()
    # Replace emails, long IDs, numbers.
    q = re.sub(r"\b[\w.%-]+@[\w.-]+\.[a-z]{2,}\b", "<EMAIL>", q)
    q = re.sub(r"\b\d{4,}\b", "<NUM>", q)  # long sequences
    q = re.sub(r"\b\d+\b", "<NUM>", q)     # any number
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _pick_rep(emb: np.ndarray, idxs: List[int]) -> int:
    # centroid + nearest by cosine similarity (emb assumed normalized)
    sub = emb[idxs]
    c = sub.mean(axis=0, keepdims=True)
    c = c / (np.linalg.norm(c) + 1e-9)
    dots = (sub @ c.T).reshape(-1)
    best_local = int(np.argmax(dots))
    return idxs[best_local]


def _cluster_embeddings(emb: np.ndarray, min_cluster_size: int, min_samples: int, seed: int) -> np.ndarray:
    # Prefer HDBSCAN if installed
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(emb)
        return labels
    except Exception:
        # Fallback: KMeans with heuristic K
        from sklearn.cluster import KMeans
        n = emb.shape[0]
        k = max(2, int(math.sqrt(n / 2)))  # crude heuristic
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(emb)
        return labels


def _answer_stability(answer_texts: List[str], embedder) -> float:
    """
    Returns a stability score in [0,1] where 1 is very stable.
    We embed answers and compute average pairwise cosine similarity approx via centroid similarity.
    """
    if not answer_texts:
        return 0.0
    texts = [a for a in answer_texts if a.strip()]
    if len(texts) < 3:
        return 0.5  # not enough evidence
    A = embedder.embed(texts)
    c = A.mean(axis=0, keepdims=True)
    c = c / (np.linalg.norm(c) + 1e-9)
    sims = (A @ c.T).reshape(-1)
    return float(np.clip(np.mean(sims), 0.0, 1.0))


@dataclass
class FAQDetector:
    name: str = "faq_detector"
    version: str = "0.1.0"

    def analyze(self, batch: TraceBatch, ctx, config: Dict[str, Any]) -> PluginResult:
        top_k = int(config.get("top_k_clusters", 20))
        min_cluster_size = int(config.get("min_cluster_size", 15))
        min_samples = int(config.get("min_samples", 5))
        include_suggestions = bool(config.get("include_suggestions", True))
        include_answer_stability = bool(config.get("include_answer_stability", True))

        # Extract fields
        views: List[TraceView] = [ctx.view_factory.make(t) for t in batch.traces]

        q_raw: List[str] = []
        trace_ids: List[str] = []
        answers: List[str] = []
        has_error: List[bool] = []
        tool_sigs: List[str] = []

        for v in views:
            q = v.query_text()
            if not q:
                continue
            q_raw.append(q)
            trace_ids.append(v.trace_id)

            ans = v.final_answer() or ""
            answers.append(ans)

            has_error.append(v.error() is not None)

            tnames = [tc.name for tc in v.tool_calls()]
            tool_sigs.append(" > ".join(tnames) if tnames else "")

        if len(q_raw) < 10:
            return PluginResult(
                metrics=[
                    MetricRecord(
                        name="faq.coverage",
                        data={
                            "trace_count_with_query": len(q_raw),
                            "note": "Not enough queries to cluster (need >= 10).",
                        },
                    )
                ]
            )

        q_norm = [_normalize_query(q) for q in q_raw]
        emb = ctx.resources.embedder.embed(q_norm)

        labels = _cluster_embeddings(
            emb,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            seed=ctx.limits.random_seed,
        )

        noise_share = float(np.mean(labels == -1))
        unique = sorted(set(int(x) for x in labels.tolist() if x != -1))

        clusters = []
        for cid in unique:
            idxs = [i for i, lab in enumerate(labels.tolist()) if int(lab) == cid]
            rep_i = _pick_rep(emb, idxs)
            count = len(idxs)
            share = count / len(labels)

            er = float(np.mean([has_error[i] for i in idxs])) if idxs else 0.0

            # Tool signature mode
            ts = [tool_sigs[i] for i in idxs if tool_sigs[i]]
            tool_mode = ""
            if ts:
                from collections import Counter
                tool_mode = Counter(ts).most_common(1)[0][0]

            # Answer stability
            stab = None
            if include_answer_stability:
                ans_texts = [answers[i] for i in idxs if answers[i].strip()]
                stab = _answer_stability(ans_texts, ctx.resources.embedder)

            clusters.append(
                {
                    "cluster_id": int(cid),
                    "count": int(count),
                    "traffic_share": float(share),
                    "rep_query": q_raw[rep_i],
                    "rep_query_norm": q_norm[rep_i],
                    "example_trace_ids": [trace_ids[i] for i in idxs[:10]],
                    "error_rate": float(er),
                    "tool_signature": tool_mode,
                    "answer_stability": stab,
                }
            )

        clusters.sort(key=lambda c: c["count"], reverse=True)
        top = clusters[:top_k]

        # Coverage metrics
        def _cov(k: int) -> float:
            return float(sum(c["traffic_share"] for c in clusters[:k])) if clusters else 0.0

        coverage = {
            "trace_count_total": len(batch.traces),
            "trace_count_with_query": len(q_raw),
            "cluster_count_total": len(clusters),
            "noise_share": noise_share,
            "traffic_top_1": _cov(1),
            "traffic_top_3": _cov(3),
            "traffic_top_5": _cov(5),
            "traffic_top_10": _cov(10),
            "traffic_top_20": _cov(20),
        }

        # Suggestions (simple rule-based for PoC)
        suggestions = []
        if include_suggestions:
            for c in top:
                recs = []
                if c["answer_stability"] is not None and c["answer_stability"] >= 0.85 and c["traffic_share"] >= 0.05:
                    recs.append("Semantic cache / template candidate (high stability, high volume).")
                if c.get("tool_signature") and c["traffic_share"] >= 0.05:
                    recs.append(f"Consider specialization: common tool path `{c['tool_signature']}`.")
                if c["error_rate"] >= 0.1:
                    recs.append("High failure rate in this FAQ cluster → prioritize reliability/RCA.")
                if not recs:
                    recs.append("Monitor; consider deflection/caching if stability increases.")
                suggestions.append({"cluster_id": c["cluster_id"], "recommendations": recs})

        # Write artifacts
        os.makedirs(ctx.output_dir, exist_ok=True)
        clusters_path = os.path.join(ctx.output_dir, "faq_clusters.json")
        with open(clusters_path, "w", encoding="utf-8") as f:
            json.dump({"coverage": coverage, "top_clusters": top, "all_clusters": clusters, "suggestions": suggestions}, f, indent=2)

        report_path = os.path.join(ctx.output_dir, "faq_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# FAQ Detector Report\n\n")
            f.write(f"- Traces analyzed: **{coverage['trace_count_total']}**\n")
            f.write(f"- Traces with query_text: **{coverage['trace_count_with_query']}**\n")
            f.write(f"- Clusters found: **{coverage['cluster_count_total']}**\n")
            f.write(f"- Noise/unclustered share: **{coverage['noise_share']:.2%}**\n\n")
            f.write("## Coverage\n\n")
            f.write(f"- Top 1 clusters cover: **{coverage['traffic_top_1']:.2%}** of clustered traffic\n")
            f.write(f"- Top 3 clusters cover: **{coverage['traffic_top_3']:.2%}** of clustered traffic\n")
            f.write(f"- Top 5 clusters cover: **{coverage['traffic_top_5']:.2%}** of clustered traffic\n")
            f.write(f"- Top 10 clusters cover: **{coverage['traffic_top_10']:.2%}** of clustered traffic\n")
            f.write(f"- Top 20 clusters cover: **{coverage['traffic_top_20']:.2%}** of clustered traffic\n\n")

            f.write("## Top FAQ clusters\n\n")
            for i, c in enumerate(top, start=1):
                f.write(f"### {i}. Cluster {c['cluster_id']} — {c['count']} traces ({c['traffic_share']:.2%})\n\n")
                f.write(f"**Representative query:** {c['rep_query']}\n\n")
                f.write(f"- Error rate: **{c['error_rate']:.2%}**\n")
                if c.get("tool_signature"):
                    f.write(f"- Common tool path: `{c['tool_signature']}`\n")
                if c.get("answer_stability") is not None:
                    f.write(f"- Answer stability: **{c['answer_stability']:.2f}** (higher is more stable)\n")
                if include_suggestions:
                    rec = next((s for s in suggestions if s["cluster_id"] == c["cluster_id"]), None)
                    if rec:
                        f.write("- Recommendations:\n")
                        for r in rec["recommendations"]:
                            f.write(f"  - {r}\n")
                f.write("\nExample trace IDs:\n\n")
                for tid in c["example_trace_ids"][:5]:
                    f.write(f"- `{tid}`\n")
                f.write("\n")

        # Trace annotations (write cluster_id on each trace)
        ann = []
        top_ids = {c["cluster_id"] for c in top}
        for tid, lab in zip(trace_ids, labels.tolist()):
            lab_i = int(lab)
            if lab_i == -1:
                continue
            ann.append(
                TraceAnnotation(
                    trace_id=tid,
                    annotations={
                        "faq.cluster_id": lab_i,
                        "faq.is_top_cluster": bool(lab_i in top_ids),
                    },
                )
            )

        ann_path = os.path.join(ctx.output_dir, "faq_trace_annotations.json")
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump([{"trace_id": a.trace_id, "annotations": a.annotations} for a in ann], f, indent=2)

        summary = {
            "top_questions": [c["rep_query"] for c in top[:3]],
            "top_coverage": {
                "top_1": coverage.get("traffic_top_1"),
                "top_5": coverage.get("traffic_top_5"),
                "top_10": coverage.get("traffic_top_10"),
            },
            "clusters": coverage.get("cluster_count_total"),
            "unclustered": coverage.get("noise_share"),
            "queries_analyzed": coverage.get("trace_count_with_query"),
        }

        summary_md = (
            f"**Top question types (representative questions):**\n"
            f"1. {top[0]['rep_query'] if len(top) > 0 else '-'}\n"
            f"2. {top[1]['rep_query'] if len(top) > 1 else '-'}\n"
            f"3. {top[2]['rep_query'] if len(top) > 2 else '-'}\n\n"
            f"- Top-5 coverage: **{coverage.get('traffic_top_5', 0.0) * 100:.1f}%**\n"
            f"- Unclustered: **{coverage.get('noise_share', 0.0) * 100:.1f}%**\n"
        )

        return PluginResult(
            metrics=[
                MetricRecord(name="faq.coverage", data=coverage),
                MetricRecord(name="faq.cluster", data={"top_clusters": top}),
            ],
            artifacts=[
                Artifact(kind="json", path=clusters_path, description="FAQ clusters bundle"),
                Artifact(kind="markdown", path=report_path, description="FAQ report"),
                Artifact(kind="json", path=ann_path, description="Per-trace FAQ annotations"),
            ],
            annotations=ann,
            summary=summary,
            summary_md=summary_md,
        )
