import os
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from collections import Counter
import numpy as np

from ..core.plugin import PluginResult, MetricRecord, Artifact, TraceAnnotation
from ..core.model import TraceBatch
from ..core.view import TraceView
from ..utils.textnorm import shorten, norm_text


_WORD = re.compile(r"[A-Za-z0-9_]+")
_SENT_SPLIT = re.compile(r"[.!?]+")

REASONING_CUES = {
    "why", "how", "explain", "analyze", "compare", "contrast", "tradeoff", "optimize",
    "prove", "derive", "evaluate", "assess", "reason", "argue", "design", "propose",
    "recommend", "strategy", "approach", "pros", "cons",
}

MULTISTEP_CUES = {
    "step", "steps", "step-by-step", "plan", "pipeline", "workflow", "procedure", "algorithm",
    "implement", "build", "create", "develop", "architect", "end-to-end",
}

CONSTRAINT_CUES = {
    "must", "should", "need", "requirements", "constraint", "constraints", "limited",
    "without", "under", "within", "at least", "at most", "exactly", "only",
}

AMBIGUITY_CUES = {
    "maybe", "possibly", "it depends", "roughly", "approximately", "suggest", "ideas", "brainstorm",
}

TECH_CUES = {
    "api", "yaml", "json", "http", "grpc", "kubernetes", "openshift", "docker", "ddp", "deepspeed",
    "vector", "embedding", "rag", "retrieval", "elasticsearch", "milvus", "llm", "mistral", "vllm",
    "latency", "throughput", "p95", "p99", "benchmark", "eval", "tracing", "otel", "mlflow",
}

CODE_TOKENS = {"{", "}", "[", "]", "(", ")", ";", "=>", "::", "def ", "class ", "import ", "SELECT", "FROM", "WHERE"}


def _count_words(s: str) -> int:
    return len(_WORD.findall(s))


def _count_sentences(s: str) -> int:
    # rough
    parts = [p.strip() for p in _SENT_SPLIT.split(s) if p.strip()]
    return max(1, len(parts))


def _count_numbers(s: str) -> int:
    return sum(ch.isdigit() for ch in s)


def _contains_any(s: str, cues: set[str]) -> int:
    sl = s.lower()
    return sum(1 for c in cues if c in sl)


def _has_code_like(s: str) -> bool:
    sl = s.lower()
    if "```" in sl:
        return True
    for tok in CODE_TOKENS:
        if tok.lower() in sl:
            return True
    # regex-ish / paths
    if "\\" in s or "/" in s or re.search(r"\b[a-zA-Z0-9_\-]+\.py\b", s):
        return True
    if re.search(r"\b[a-zA-Z0-9_\-]+://", s):
        return True
    return False


def _listiness(s: str) -> float:
    # bullets/lists usually mean multiple requirements
    lines = s.splitlines()
    if len(lines) <= 1:
        return 0.0
    bullet = sum(1 for ln in lines if ln.strip().startswith(("-", "*", "1.", "2.", "3.")))
    return min(1.0, bullet / max(1, len(lines)))


def _acronym_density(s: str) -> float:
    # crude: many ALLCAPS tokens -> technical density
    tokens = _WORD.findall(s)
    if not tokens:
        return 0.0
    caps = sum(1 for t in tokens if len(t) >= 2 and t.isupper())
    return caps / len(tokens)


def _feature_score(query: str) -> Tuple[float, Dict[str, float]]:
    """
    Advanced heuristic complexity score in [0,1] with feature breakdown.
    """
    q = query.strip()
    qn = norm_text(q)

    w = _count_words(q)
    sents = _count_sentences(q)
    chars = len(q)

    # Normalize length
    # word length saturates around ~60 words
    len_score = min(1.0, w / 60.0)

    # Structure
    clause_markers = sum(q.count(x) for x in [",", ";", " and ", " or ", " but ", " however "])
    struct_score = min(1.0, (clause_markers / 6.0) + (sents - 1) * 0.15)

    # Cues
    reasoning = min(1.0, _contains_any(q, REASONING_CUES) / 4.0)
    multistep = min(1.0, _contains_any(q, MULTISTEP_CUES) / 3.0)
    constraints = min(1.0, _contains_any(q, CONSTRAINT_CUES) / 4.0)
    ambiguity = min(1.0, _contains_any(q, AMBIGUITY_CUES) / 3.0)

    # Technical / domain
    tech = min(1.0, _contains_any(q, TECH_CUES) / 4.0)
    acr = min(1.0, _acronym_density(q) * 4.0)

    # Quant/filters/code
    nums = min(1.0, _count_numbers(q) / 25.0)
    code = 1.0 if _has_code_like(q) else 0.0
    listy = _listiness(q)

    # Combine with calibrated weights (sum ~1)
    score = (
        0.18 * len_score +
        0.14 * struct_score +
        0.16 * reasoning +
        0.12 * multistep +
        0.12 * constraints +
        0.10 * tech +
        0.05 * acr +
        0.05 * nums +
        0.05 * code +
        0.03 * listy
    )

    # Small bump for very long prompts (characters)
    if chars > 600:
        score += 0.05
    if chars > 1500:
        score += 0.05

    score = float(np.clip(score, 0.0, 1.0))

    feats = {
        "len": float(len_score),
        "structure": float(struct_score),
        "reasoning": float(reasoning),
        "multistep": float(multistep),
        "constraints": float(constraints),
        "tech": float(tech),
        "acronyms": float(acr),
        "numbers": float(nums),
        "code": float(code),
        "listiness": float(listy),
    }
    return score, feats


def _bucket(score: float, thresholds: Dict[str, float]) -> str:
    s = float(score)
    if s <= thresholds["simple_max"]:
        return "simple"
    if s <= thresholds["medium_max"]:
        return "medium"
    if s <= thresholds["complex_max"]:
        return "complex"
    return "very_complex"


@dataclass
class QueryComplexityAnalyzer:
    name: str = "query_complexity_analyzer"
    version: str = "0.1.0"

    def analyze(self, batch: TraceBatch, ctx, config: Dict[str, Any]) -> PluginResult:
        thresholds = {
            "simple_max": float(config.get("simple_max", 0.22)),
            "medium_max": float(config.get("medium_max", 0.50)),
            "complex_max": float(config.get("complex_max", 0.78)),
        }

        slm_candidate_pct = float(config.get("slm_candidate_pct", 0.95))
        frontier_pct = float(config.get("frontier_pct", 0.10))

        views: List[TraceView] = [ctx.view_factory.make(t) for t in batch.traces]
        texts: List[str] = []
        trace_ids: List[str] = []
        timestamps: List[int] = []

        for t, v in zip(batch.traces, views):
            text = v.query_text()
            text = (text or "").strip()
            if not text:
                continue
            texts.append(text)
            trace_ids.append(v.trace_id)
            timestamps.append(int(t.start_time_ms))

        n = len(texts)
        if n == 0:
            summary_md = f"ℹ️ No queries available for complexity scoring."
            return PluginResult(
                metrics=[MetricRecord(name="complexity.summary", data={"total": 0})],
                artifacts=[],
                summary_md=summary_md,
            )

        scores: List[float] = []
        feats_list: List[Dict[str, float]] = []
        buckets: List[str] = []

        for q in texts:
            s, feats = _feature_score(q)
            scores.append(s)
            feats_list.append(feats)
            buckets.append(_bucket(s, thresholds))

        bucket_counts = Counter(buckets)
        frac = {k: bucket_counts.get(k, 0) / n for k in ["simple", "medium", "complex", "very_complex"]}

        arr = np.asarray(scores, dtype=float)
        stats = {
            "total": n,
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

        simpleish = frac["simple"] + frac["medium"]
        recs: List[str] = []
        if simpleish >= slm_candidate_pct and frac["very_complex"] <= frontier_pct:
            recs.append(
                f"✅ **{simpleish*100:.1f}%** of queries are **simple/medium** → consider routing most traffic to a cheap SLM and escalate only hard cases."
            )
        elif simpleish >= 0.80:
            recs.append(
                f"⚙️ **{simpleish*100:.1f}%** of queries are **simple/medium** → consider a cascaded router (SLM first, frontier on high complexity)."
            )
        else:
            recs.append(
                f"ℹ️ Only **{simpleish*100:.1f}%** are simple/medium → a strong model is needed for a large fraction of traffic."
            )

        if frac["very_complex"] >= 0.15:
            recs.append(f"⚠️ **{frac['very_complex']*100:.1f}%** are **very complex** → keep a strong model for those.")
        if frac["simple"] >= 0.50:
            recs.append("💡 Many queries are **very simple** → templates/deflection are likely effective (use FAQ detector).")

        # Top hard/easy examples
        idx_sorted_hard = sorted(range(n), key=lambda i: scores[i], reverse=True)
        idx_sorted_easy = sorted(range(n), key=lambda i: scores[i])

        top_hard = [{"trace_id": trace_ids[i], "score": scores[i], "bucket": buckets[i], "query": texts[i]} for i in idx_sorted_hard[:10]]
        top_easy = [{"trace_id": trace_ids[i], "score": scores[i], "bucket": buckets[i], "query": texts[i]} for i in idx_sorted_easy[:10]]

        # Aggregate feature importance (mean feature values)
        feat_means = {k: float(np.mean([f[k] for f in feats_list])) for k in feats_list[0].keys()}
        top_feat = sorted(feat_means.items(), key=lambda x: x[1], reverse=True)[:4]

        # Annotations
        annotations: List[TraceAnnotation] = []
        per_trace = []
        for i in range(n):
            annotations.append(
                TraceAnnotation(
                    trace_id=trace_ids[i],
                    annotations={
                        "complexity.score": float(scores[i]),
                        "complexity.bucket": buckets[i],
                        "complexity.method": "heuristic_v2",
                    },
                )
            )
            per_trace.append({
                "trace_id": trace_ids[i],
                "timestamp_ms": timestamps[i],
                "score": float(scores[i]),
                "bucket": buckets[i],
                "features": feats_list[i],
                "query": texts[i],
            })

        summary_md = (
            "**Query complexity distribution** (heuristic)\n\n"
            f"- Simple: **{frac['simple']*100:.1f}%**\n"
            f"- Medium: **{frac['medium']*100:.1f}%**\n"
            f"- Complex: **{frac['complex']*100:.1f}%**\n"
            f"- Very complex: **{frac['very_complex']*100:.1f}%**\n\n"
            f"**Stats:** mean **{stats['mean']:.2f}**, p90 **{stats['p90']:.2f}**, p95 **{stats['p95']:.2f}**\n\n"
        )
        summary_md += "**Top drivers (avg feature signals):**\n"
        for k, v in top_feat:
            summary_md += f"- {k}: **{v:.2f}**\n"
        summary_md += "\n**Recommendations:**\n" + "\n".join([f"- {r}" for r in recs[:3]]) + "\n"

        # Write artifacts
        os.makedirs(ctx.output_dir, exist_ok=True)
        json_path = os.path.join(ctx.output_dir, "complexity.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": {
                        "thresholds": thresholds,
                        "bucket_counts": dict(bucket_counts),
                        "bucket_fractions": frac,
                        "stats": stats,
                        "recommendations": recs,
                        "feature_means": feat_means,
                        "method": "heuristic_v2",
                    },
                    "top_hard": top_hard,
                    "top_easy": top_easy,
                    "per_trace": per_trace,
                },
                f,
                indent=2,
            )

        md_path = os.path.join(ctx.output_dir, "complexity_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Query Complexity Analysis (Heuristic)\n\n")
            f.write(f"- Queries analyzed: **{n}**\n\n")
            f.write("## Distribution\n\n")
            f.write(f"- Simple: **{frac['simple']*100:.1f}%** ({bucket_counts.get('simple', 0)})\n")
            f.write(f"- Medium: **{frac['medium']*100:.1f}%** ({bucket_counts.get('medium', 0)})\n")
            f.write(f"- Complex: **{frac['complex']*100:.1f}%** ({bucket_counts.get('complex', 0)})\n")
            f.write(f"- Very complex: **{frac['very_complex']*100:.1f}%** ({bucket_counts.get('very_complex', 0)})\n\n")

            f.write("## Feature drivers (mean signals)\n\n")
            for k, v in sorted(feat_means.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {k}: {v:.2f}\n")
            f.write("\n## Recommendations\n\n")
            for r in recs:
                f.write(f"- {r}\n")

            f.write("\n## Most complex examples\n\n")
            for x in top_hard[:5]:
                f.write(f"- `{x['trace_id']}` (score {x['score']:.2f}, {x['bucket']}): {shorten(x['query'], 220)}\n")

            f.write("\n## Least complex examples\n\n")
            for x in top_easy[:5]:
                f.write(f"- `{x['trace_id']}` (score {x['score']:.2f}, {x['bucket']}): {shorten(x['query'], 220)}\n")

        ann_path = os.path.join(ctx.output_dir, "complexity_trace_annotations.json")
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump([{"trace_id": a.trace_id, "annotations": a.annotations} for a in annotations], f, indent=2)

        return PluginResult(
            metrics=[
                MetricRecord(name="complexity.stats", data=stats),
                MetricRecord(name="complexity.bucket_counts", data=dict(bucket_counts)),
                MetricRecord(name="complexity.bucket_fractions", data=frac),
            ],
            artifacts=[
                Artifact(kind="markdown", path=md_path, description="Query complexity report"),
                Artifact(kind="json", path=json_path, description="Query complexity JSON"),
                Artifact(kind="json", path=ann_path, description="Per-trace complexity annotations"),
            ],
            annotations=annotations,
            summary_md=summary_md,
            summary={"bucket_fractions": frac, "stats": stats},
        )
