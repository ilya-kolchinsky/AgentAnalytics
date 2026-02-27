import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List
from collections import Counter
import itertools

from ..core.plugin import PluginResult, MetricRecord, Artifact
from ..core.model import TraceBatch
from ..core.view import TraceView


def _canonical_tool_sequence(tool_calls) -> List[str]:
    # Keep order; drop empty; normalize to string
    seq = []
    for tc in tool_calls:
        name = str(tc.name).strip()
        if name:
            seq.append(name)
    return seq


def _ngram_counts(seqs: List[List[str]], n: int) -> Counter:
    c = Counter()
    for s in seqs:
        if len(s) < n:
            continue
        for i in range(len(s) - n + 1):
            c[tuple(s[i : i + n])] += 1
    return c


def _apriori_frequent_itemsets(
    transactions: List[frozenset[str]],
    *,
    min_support: float,
    max_size: int = 4,
) -> Dict[frozenset[str], float]:
    """
    Simple Apriori over tool sets.
    Returns itemset -> support (fraction of transactions containing it).
    """
    n_tx = len(transactions)
    if n_tx == 0:
        return {}

    # Count singletons
    item_counts = Counter()
    for t in transactions:
        item_counts.update(t)

    def support(count: int) -> float:
        return count / n_tx

    # L1
    L = {frozenset([i]): support(c) for i, c in item_counts.items() if support(c) >= min_support}
    all_freq = dict(L)

    k = 2
    prev_L = set(L.keys())
    while prev_L and k <= max_size:
        # candidate generation: join sets that share k-2 items
        prev_list = list(prev_L)
        candidates = set()
        for a, b in itertools.combinations(prev_list, 2):
            u = a | b
            if len(u) == k:
                # prune: all (k-1)-subsets must be frequent
                ok = True
                for sub in itertools.combinations(u, k - 1):
                    if frozenset(sub) not in prev_L:
                        ok = False
                        break
                if ok:
                    candidates.add(u)

        if not candidates:
            break

        # count candidates
        cand_counts = Counter()
        for t in transactions:
            for cset in candidates:
                if cset.issubset(t):
                    cand_counts[cset] += 1

        Lk = {cset: support(cnt) for cset, cnt in cand_counts.items() if support(cnt) >= min_support}
        all_freq.update(Lk)
        prev_L = set(Lk.keys())
        k += 1

    return all_freq


def _association_rules(
    freq_itemsets: Dict[frozenset[str], float],
    *,
    min_confidence: float,
    min_lift: float,
) -> List[Dict[str, Any]]:
    """
    Generate rules A -> B from frequent itemsets.
    Uses support(A∪B), support(A), support(B) to compute confidence and lift.
    """
    rules = []
    # Precompute supports for quick lookup
    supp = freq_itemsets

    for itemset, sup_ab in freq_itemsets.items():
        if len(itemset) < 2:
            continue

        items = list(itemset)
        # all non-empty proper subsets as antecedents
        for r in range(1, len(items)):
            for antecedent_tuple in itertools.combinations(items, r):
                A = frozenset(antecedent_tuple)
                B = itemset - A
                if not B:
                    continue
                sup_a = supp.get(A)
                sup_b = supp.get(B)
                if sup_a is None or sup_b is None:
                    continue
                conf = sup_ab / sup_a if sup_a > 0 else 0.0
                lift = conf / sup_b if sup_b > 0 else 0.0

                if conf >= min_confidence and lift >= min_lift:
                    rules.append({
                        "antecedent": sorted(list(A)),
                        "consequent": sorted(list(B)),
                        "support": sup_ab,
                        "confidence": conf,
                        "lift": lift,
                    })

    # Sort by lift, then confidence, then support
    rules.sort(key=lambda x: (x["lift"], x["confidence"], x["support"]), reverse=True)
    return rules


@dataclass
class ToolUsePatterns:
    name: str = "tool_use_patterns"
    version: str = "0.1.0"

    def analyze(self, batch: TraceBatch, ctx, config: Dict[str, Any]) -> PluginResult:
        # --- config ---
        min_support = float(config.get("min_support", 0.03))  # fraction of traces
        max_itemset_size = int(config.get("max_itemset_size", 4))
        min_confidence = float(config.get("min_confidence", 0.5))
        min_lift = float(config.get("min_lift", 1.2))

        max_ngram_n = int(config.get("max_ngram_n", 4))
        min_ngram_count = int(config.get("min_ngram_count", 5))

        top_k_itemsets = int(config.get("top_k_itemsets", 25))
        top_k_rules = int(config.get("top_k_rules", 25))
        top_k_ngrams = int(config.get("top_k_ngrams", 25))

        # --- extract tool sequences ---
        views: List[TraceView] = [ctx.view_factory.make(t) for t in batch.traces]

        seqs: List[List[str]] = []
        trace_ids: List[str] = []
        for v in views:
            seq = _canonical_tool_sequence(v.tool_calls())
            if not seq:
                continue
            seqs.append(seq)
            trace_ids.append(v.trace_id)

        n_traces_with_tools = len(seqs)
        if n_traces_with_tools == 0:
            summary_md = "ℹ️ No tool calls found in this window, so no tool-use patterns can be mined."
            return PluginResult(
                metrics=[MetricRecord(name="tool_patterns.summary", data={"traces_with_tools": 0})],
                artifacts=[],
                summary_md=summary_md,
            )

        # --- transactions for itemset mining (unique tools per trace) ---
        transactions = [frozenset(s) for s in seqs]

        # --- frequent itemsets ---
        freq_itemsets = _apriori_frequent_itemsets(
            transactions,
            min_support=min_support,
            max_size=max_itemset_size,
        )
        # sort itemsets by support then size
        itemsets_sorted = sorted(
            [{"items": sorted(list(k)), "size": len(k), "support": v} for k, v in freq_itemsets.items() if len(k) >= 2],
            key=lambda x: (x["support"], x["size"]),
            reverse=True,
        )
        top_itemsets = itemsets_sorted[:top_k_itemsets]

        # --- association rules ---
        rules = _association_rules(freq_itemsets, min_confidence=min_confidence, min_lift=min_lift)
        top_rules = rules[:top_k_rules]

        # --- sequence n-grams ---
        ngram_results = []
        for n in range(2, max_ngram_n + 1):
            counts = _ngram_counts(seqs, n)
            for ng, c in counts.items():
                if c >= min_ngram_count:
                    ngram_results.append({
                        "ngram": list(ng),
                        "n": n,
                        "count": c,
                        "support": c / n_traces_with_tools,  # support over traces-with-tools (approx)
                    })
        ngram_results.sort(key=lambda x: (x["count"], x["n"]), reverse=True)
        top_ngrams = ngram_results[:top_k_ngrams]

        # --- simple recommendations for meta-tools ---
        # Prefer strong sequences (n-grams) and strong rules A->B with high lift.
        recs: List[str] = []
        if top_ngrams:
            best = top_ngrams[0]
            recs.append(
                f"Consider a meta-tool for sequence: `{ ' -> '.join(best['ngram']) }` "
                f"(count {best['count']}, support ~{best['support']*100:.1f}%)."
            )
        if top_rules:
            r0 = top_rules[0]
            recs.append(
                f"Strong coupling rule: `{'+'.join(r0['antecedent'])} -> {'+'.join(r0['consequent'])}` "
                f"(conf {r0['confidence']*100:.1f}%, lift {r0['lift']:.2f}, support {r0['support']*100:.1f}%)."
            )

        # If tools are many and patterns show a small core, highlight it
        all_tools = sorted(set(itertools.chain.from_iterable(seqs)))
        tool_counts = Counter(itertools.chain.from_iterable(seqs))
        top_tools = [t for t, _ in tool_counts.most_common(10)]
        core_share = sum(tool_counts[t] for t in top_tools) / max(1, sum(tool_counts.values()))
        if core_share >= 0.75:
            recs.append(
                f"Top-10 tools account for {core_share*100:.1f}% of tool calls → consider consolidating frequent chains into meta-tools."
            )

        # --- summary_md ---
        summary_md = (
            f"**Tool-use patterns mined** (from {n_traces_with_tools} traces with tool calls)\n\n"
            f"- Distinct tools seen: **{len(all_tools)}**\n"
            f"- Frequent itemsets (≥2 tools): **{len(itemsets_sorted)}**\n"
            f"- Association rules: **{len(rules)}** (min_conf {min_confidence:.2f}, min_lift {min_lift:.2f})\n"
            f"- Frequent n-grams: **{len(ngram_results)}** (min_count {min_ngram_count})\n\n"
        )
        if top_ngrams:
            summary_md += "**Top sequences:**\n"
            for x in top_ngrams[:3]:
                summary_md += f"- `{ ' -> '.join(x['ngram']) }` — {x['count']} traces\n"
            summary_md += "\n"
        if top_rules:
            summary_md += "**Top associations:**\n"
            for r in top_rules[:3]:
                summary_md += (
                    f"- `{'+'.join(r['antecedent'])} -> {'+'.join(r['consequent'])}` "
                    f"(conf {r['confidence']*100:.1f}%, lift {r['lift']:.2f})\n"
                )
            summary_md += "\n"
        if recs:
            summary_md += "**Recommendations:**\n" + "\n".join([f"- {r}" for r in recs[:3]]) + "\n"

        summary = {
            "traces_with_tools": n_traces_with_tools,
            "distinct_tools": len(all_tools),
            "itemsets": len(itemsets_sorted),
            "rules": len(rules),
            "ngrams": len(ngram_results),
            "top_tools": top_tools,
        }

        # --- artifacts ---
        os.makedirs(ctx.output_dir, exist_ok=True)
        json_path = os.path.join(ctx.output_dir, "tool_use_patterns.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "config": {
                        "min_support": min_support,
                        "max_itemset_size": max_itemset_size,
                        "min_confidence": min_confidence,
                        "min_lift": min_lift,
                        "max_ngram_n": max_ngram_n,
                        "min_ngram_count": min_ngram_count,
                    },
                    "top_tools": [{"tool": t, "count": tool_counts[t]} for t in top_tools],
                    "top_itemsets": top_itemsets,
                    "top_rules": top_rules,
                    "top_ngrams": top_ngrams,
                    "all_tools": all_tools,
                },
                f,
                indent=2,
            )

        md_path = os.path.join(ctx.output_dir, "tool_use_patterns_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Tool Use Patterns\n\n")
            f.write(f"- Traces with tools: **{n_traces_with_tools}**\n")
            f.write(f"- Distinct tools: **{len(all_tools)}**\n\n")

            f.write("## Top tools\n\n")
            for t in top_tools:
                f.write(f"- `{t}`: {tool_counts[t]}\n")

            f.write("\n## Frequent tool sets (itemsets)\n\n")
            if not top_itemsets:
                f.write("No frequent itemsets met the minimum support threshold.\n")
            else:
                for it in top_itemsets:
                    f.write(f"- {it['items']} — support **{it['support']*100:.1f}%**\n")

            f.write("\n## Association rules\n\n")
            if not top_rules:
                f.write("No rules met the confidence/lift thresholds.\n")
            else:
                for r in top_rules:
                    f.write(
                        f"- `{'+'.join(r['antecedent'])} -> {'+'.join(r['consequent'])}` "
                        f"(support {r['support']*100:.1f}%, conf {r['confidence']*100:.1f}%, lift {r['lift']:.2f})\n"
                    )

            f.write("\n## Frequent sequences (n-grams)\n\n")
            if not top_ngrams:
                f.write("No n-grams met the minimum count threshold.\n")
            else:
                for x in top_ngrams:
                    f.write(f"- `{ ' -> '.join(x['ngram']) }` — {x['count']} traces (support ~{x['support']*100:.1f}%)\n")

            f.write("\n## Recommendations\n\n")
            if not recs:
                f.write("No strong consolidation recommendations found with current thresholds.\n")
            else:
                for r in recs:
                    f.write(f"- {r}\n")

        return PluginResult(
            metrics=[
                MetricRecord(name="tool_patterns.summary", data=summary),
                MetricRecord(name="tool_patterns.top_rules", data={"top_rules": top_rules[:10]}),
                MetricRecord(name="tool_patterns.top_ngrams", data={"top_ngrams": top_ngrams[:10]}),
            ],
            artifacts=[
                Artifact(kind="markdown", path=md_path, description="Tool use patterns report"),
                Artifact(kind="json", path=json_path, description="Tool use patterns JSON"),
            ],
            summary_md=summary_md,
            summary=summary,
        )
