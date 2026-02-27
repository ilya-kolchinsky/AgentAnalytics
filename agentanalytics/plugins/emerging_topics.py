import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List
from collections import defaultdict

from ..core.plugin import PluginResult, MetricRecord, Artifact
from ..core.model import TraceBatch
from ..core.view import TraceView
from ..utils.textnorm import norm_numbers
from ..utils.clustering import cluster_embeddings, pick_representative_index


def _split_by_time(times: List[int]) -> List[int]:
    return [i for i, _ in sorted(enumerate(times), key=lambda x: x[1])]


@dataclass
class EmergingTopicsDetector:
    name: str = "emerging_topics_detector"
    version: str = "0.1.0"

    def analyze(self, batch: TraceBatch, ctx, config: Dict[str, Any]) -> PluginResult:
        top_k = int(config.get("top_k_emerging", 15))
        min_cluster_size = int(config.get("min_cluster_size", 15))
        min_samples = int(config.get("min_samples", 5))

        # NEW knobs to make it distinct from FAQ
        min_late_count = int(config.get("min_late_count", max(5, min_cluster_size // 2)))
        min_growth = float(config.get("min_growth", 1.8))  # late/early ratio threshold
        require_new_or_growing = bool(config.get("require_new_or_growing", True))

        views: List[TraceView] = [ctx.view_factory.make(t) for t in batch.traces]

        q_raw: List[str] = []
        q_norm: List[str] = []
        trace_ids: List[str] = []
        times: List[int] = []

        for t, v in zip(batch.traces, views):
            q = v.query_text()
            if not q:
                continue
            q_raw.append(q)
            q_norm.append(norm_numbers(q))
            trace_ids.append(v.trace_id)
            times.append(int(t.start_time_ms))

        if len(q_raw) < 20:
            summary = {
                "headline": "Not enough queries for trend detection",
                "kpis": {"queries_analyzed": len(q_raw)},
                "bullets": ["Need at least 20 queries to detect emerging topics reliably."]
            }
            return PluginResult(
                metrics=[MetricRecord(name="trends.summary", data=summary)],
                artifacts=[],
                summary=summary,
            )

        emb = ctx.resources.embedder.embed(q_norm)
        cr = cluster_embeddings(emb, min_cluster_size=min_cluster_size, min_samples=min_samples, seed=ctx.limits.random_seed)
        labels = cr.labels

        order = _split_by_time(times)
        mid = len(order) // 2
        early_idx = set(order[:mid])
        late_idx = set(order[mid:])

        cluster_counts = defaultdict(lambda: {"early": 0, "late": 0, "total": 0, "idxs": []})
        for i, lab in enumerate(labels.tolist()):
            lab = int(lab)
            if lab == -1:
                continue
            cc = cluster_counts[lab]
            cc["total"] += 1
            cc["idxs"].append(i)
            if i in early_idx:
                cc["early"] += 1
            else:
                cc["late"] += 1

        clusters = []
        for cid, cc in cluster_counts.items():
            idxs = cc["idxs"]
            rep_i = pick_representative_index(emb, idxs)
            early = cc["early"]
            late = cc["late"]
            total = cc["total"]

            growth = (late + 1.0) / (early + 1.0)
            is_new = early == 0 and late >= min_late_count
            is_growing = (late >= min_late_count) and (growth >= min_growth)

            clusters.append({
                "cluster_id": int(cid),
                "total": int(total),
                "early": int(early),
                "late": int(late),
                "growth": float(growth),
                "is_new": bool(is_new),
                "is_growing": bool(is_growing),
                "rep_query": q_raw[rep_i],
                "example_trace_ids": [trace_ids[i] for i in idxs[:8]],
            })

        # Focus only on emerging (distinct from FAQ)
        if require_new_or_growing:
            emerging = [c for c in clusters if c["is_new"] or c["is_growing"]]
        else:
            emerging = clusters

        emerging.sort(key=lambda c: (c["is_new"], c["growth"], c["late"]), reverse=True)
        emerging = emerging[:top_k]

        if not emerging:
            summary = {
                "headline": "No emerging topics detected",
                "kpis": {
                    "queries_analyzed": len(q_raw),
                    "clusters": len(clusters),
                    "noise_share": cr.noise_share,
                },
                "bullets": [
                    "No clusters met the 'new' or 'growing' threshold in this window.",
                    "Try lowering min_growth/min_late_count, or increase the analysis timeframe."
                ],
            }
        else:
            summary = {
                "headline": f"{len(emerging)} emerging topics detected",
                "kpis": {
                    "queries_analyzed": len(q_raw),
                    "clusters": len(clusters),
                    "noise_share": cr.noise_share,
                    "emerging": len(emerging),
                },
                "bullets": [f"Top emerging: {emerging[0]['rep_query']}"] + (
                    [f"Next: {emerging[1]['rep_query']}"] if len(emerging) > 1 else []
                ),
                "top_items": [c["rep_query"] for c in emerging[:3]],
            }

        os.makedirs(ctx.output_dir, exist_ok=True)
        json_path = os.path.join(ctx.output_dir, "emerging_topics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "thresholds": {
                    "min_late_count": min_late_count,
                    "min_growth": min_growth,
                    "require_new_or_growing": require_new_or_growing,
                },
                "emerging": emerging,
                "all_clusters": clusters,
            }, f, indent=2)

        md_path = os.path.join(ctx.output_dir, "emerging_topics_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Emerging Topics & Trend Shifts\n\n")
            f.write(f"- Queries analyzed: **{len(q_raw)}**\n")
            f.write(f"- Clusters: **{len(clusters)}**\n")
            f.write(f"- Unclustered/noise share: **{cr.noise_share:.2%}**\n\n")
            f.write("This report focuses only on **new or rapidly growing** query clusters (distinct from FAQ frequency).\n\n")
            f.write("## Thresholds\n\n")
            f.write(f"- min_late_count: **{min_late_count}**\n")
            f.write(f"- min_growth (late/early): **{min_growth:.2f}×**\n\n")

            if not emerging:
                f.write("✅ No emerging clusters met the thresholds in this window.\n")
            else:
                f.write("## Emerging clusters\n\n")
                for i, c in enumerate(emerging, 1):
                    tag = "NEW" if c["is_new"] else f"{c['growth']:.2f}×"
                    f.write(f"### {i}. Cluster {c['cluster_id']} — {tag}\n\n")
                    f.write(f"**Representative query:** {c['rep_query']}\n\n")
                    f.write(f"- Early count: **{c['early']}**\n")
                    f.write(f"- Late count: **{c['late']}**\n")
                    f.write(f"- Total: **{c['total']}**\n\n")
                    if c["example_trace_ids"]:
                        f.write("Example trace IDs:\n")
                        for tid in c["example_trace_ids"][:5]:
                            f.write(f"- `{tid}`\n")
                        f.write("\n")

        if not emerging:
            summary_md = (
                "✅ **No emerging topics detected**\n\n"
                f"- Queries analyzed: **{len(q_raw)}**\n"
                f"- Try lowering thresholds or increasing timeframe.\n"
            )
            lines = "\n".join(
                [f"- {('NEW' if c['is_new'] else c['growth'])} — {c['rep_query']}" for c in emerging[:3]]
            )
            summary_md = (
                    f"📈 **{len(emerging)} emerging topics detected**\n\n"
                    f"{lines}\n"
                )

        return PluginResult(
            metrics=[
                MetricRecord(name="trends.summary", data=summary),
                MetricRecord(name="trends.emerging", data={"emerging": emerging}),
            ],
            artifacts=[
                Artifact(kind="markdown", path=md_path, description="Emerging topics report"),
                Artifact(kind="json", path=json_path, description="Emerging topics JSON"),
            ],
            summary=summary,
            summary_md=summary_md,
        )
