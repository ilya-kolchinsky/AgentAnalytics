from typing import Dict, List

from agentanalytics.core.plugin_meta import PluginMeta, PluginParam
from .faq_detector import FAQDetector
from .tool_reliability import ToolReliabilityAuditor
from .emerging_topics import EmergingTopicsDetector
from .sentiment import SentimentAnalyzer
from .query_complexity import QueryComplexityAnalyzer
from .tool_use_patterns import ToolUsePatterns


_REGISTRY = {
    "faq_detector": FAQDetector(),
    "tool_reliability_auditor": ToolReliabilityAuditor(),
    "emerging_topics_detector": EmergingTopicsDetector(),
    "sentiment_analyzer": SentimentAnalyzer(),
    "query_complexity_analyzer": QueryComplexityAnalyzer(),
    "tool_use_patterns": ToolUsePatterns(),
}

# Metadata (framework-owned)
_META: Dict[str, PluginMeta] = {}

_META["faq_detector"] = PluginMeta(
    name="faq_detector",
    version=_REGISTRY["faq_detector"].version,
    title="Frequently Asked Questions Detector",
    description="Semantically clusters user queries and surfaces the most frequent question types.",
    requires=["query_text"],
    params=[
        PluginParam(
            key="top_k_clusters",
            type="int",
            title="Top K clusters",
            description="How many top clusters to show in the report.",
            default=20,
            min=1,
            max=200,
        ),
        PluginParam(
            key="min_cluster_size",
            type="int",
            title="Min cluster size",
            description="HDBSCAN: minimum number of items in a cluster.",
            default=15,
            min=2,
            max=1000,
        ),
        PluginParam(
            key="min_samples",
            type="int",
            title="Min samples",
            description="HDBSCAN: higher values make clustering more conservative.",
            default=5,
            min=1,
            max=1000,
        ),
        PluginParam(
            key="include_suggestions",
            type="bool",
            title="Include suggestions",
            description="Add rule-based optimization suggestions to the report.",
            default=True,
        ),
        PluginParam(
            key="include_answer_stability",
            type="bool",
            title="Include answer stability",
            description="Estimate within-cluster answer stability (useful for caching/deflection decisions).",
            default=True,
        ),
    ],
)

_META["tool_reliability_auditor"] = PluginMeta(
    name="tool_reliability_auditor",
    version=_REGISTRY["tool_reliability_auditor"].version,
    title="Tool Call Reliability Auditor",
    description="Ranks tools by failures, invalid arguments, and common error signatures; outputs actionable reliability insights.",
    requires=["tool_calls"],
    params=[
        PluginParam(
            key="top_k_tools",
            type="int",
            title="Top K tools",
            description="How many tools to include in the report, ranked by errors (then by volume).",
            default=20,
            min=1,
            max=200,
        ),
        PluginParam(
            key="include_examples",
            type="bool",
            title="Include example traces",
            description="If enabled, include example trace IDs for each tool to speed up debugging.",
            default=True,
        ),
        PluginParam(
            key="max_examples",
            type="int",
            title="Max examples per tool",
            description="Maximum number of example trace IDs to include per tool (only if examples are enabled).",
            default=5,
            min=1,
            max=20,
        ),
    ],
)

_META["emerging_topics_detector"] = PluginMeta(
    name="emerging_topics_detector",
    version=_REGISTRY["emerging_topics_detector"].version,
    title="Emerging Topics & Trend Shifts",
    description="Detects new or rapidly growing query clusters by comparing earlier vs later traffic in the analysis window.",
    requires=["query_text"],
    params=[
        PluginParam(
            key="top_k_emerging",
            type="int",
            title="Top K emerging clusters",
            description="How many emerging clusters to include in the report, ranked by novelty/growth.",
            default=15,
            min=1,
            max=200,
        ),
        PluginParam(
            key="min_cluster_size",
            type="int",
            title="Min cluster size",
            description="Clustering: minimum number of queries required to form a cluster.",
            default=15,
            min=2,
            max=1000,
        ),
        PluginParam(
            key="min_samples",
            type="int",
            title="Min samples",
            description="Clustering: higher values make clustering more conservative (more noise, fewer clusters).",
            default=5,
            min=1,
            max=1000,
        ),
    ],
)

_META["sentiment_analyzer"] = PluginMeta(
    name="sentiment_analyzer",
    version=_REGISTRY["sentiment_analyzer"].version,
    title="Sentiment / Customer Satisfaction Analyzer",
    description="Estimates per-trace sentiment and dissatisfaction signals from user text, then reports distributions and top negative drivers.",
    requires=["query_text"],
    params=[
        PluginParam(
          key="model_id",
          type="str",
          title="HF model id",
          description="Hugging Face model id for sentiment analysis (lightweight default is DistilBERT SST-2).",
          default="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        ),
        PluginParam(
            key="source",
            type="enum",
            title="Text source",
            description="Which text field to score for sentiment.",
            default="query_text",
            enum=["query_text", "final_answer"],
        ),
        PluginParam(
            key="top_k_negative",
            type="int",
            title="Top K negative examples",
            description="How many most-negative examples to include in the report.",
            default=15,
            min=3,
            max=200,
        ),
        PluginParam(
            key="include_negative_clustering",
            type="bool",
            title="Cluster negative themes",
            description="If enabled, cluster negative examples to surface common dissatisfaction themes.",
            default=True,
        ),
        PluginParam(
            key="min_cluster_size",
            type="int",
            title="Min cluster size",
            description="Clustering: minimum number of negative examples required to form a cluster theme.",
            default=10,
            min=2,
            max=1000,
        ),
        PluginParam(
            key="min_samples",
            type="int",
            title="Min samples",
            description="Clustering: higher values make clustering more conservative (more noise).",
            default=3,
            min=1,
            max=1000,
        ),
    ],
)

_META["query_complexity_analyzer"] = PluginMeta(
    name="query_complexity_analyzer",
    version=_REGISTRY["query_complexity_analyzer"].version,
    title="Query Complexity Analyzer",
    description="Scores query complexity in [0,1] using a lightweight estimator model and reports bucket distributions and routing recommendations.",
    requires=["query_text"],
    params=[
        PluginParam(
            key="simple_max",
            type="float",
            title="Simple max",
            description="Upper bound for the 'simple' bucket (<= simple_max).",
            default=0.25,
            min=0.0,
            max=1.0,
        ),
        PluginParam(
            key="medium_max",
            type="float",
            title="Medium max",
            description="Upper bound for the 'medium' bucket (<= medium_max).",
            default=0.55,
            min=0.0,
            max=1.0,
        ),
        PluginParam(
            key="complex_max",
            type="float",
            title="Complex max",
            description="Upper bound for the 'complex' bucket (<= complex_max). Above this is 'very_complex'.",
            default=0.80,
            min=0.0,
            max=1.0,
        ),
        PluginParam(
            key="slm_candidate_pct",
            type="float",
            title="SLM candidate threshold",
            description="If (simple+medium) fraction exceeds this, recommend routing most traffic to an SLM.",
            default=0.95,
            min=0.0,
            max=1.0,
        ),
        PluginParam(
            key="frontier_pct",
            type="float",
            title="Frontier necessity threshold",
            description="If very_complex fraction exceeds this, recommend keeping a strong model available for that slice.",
            default=0.10,
            min=0.0,
            max=1.0,
        ),
    ],
)

_META["tool_use_patterns"] = PluginMeta(
    name="tool_use_patterns",
    version=_REGISTRY["tool_use_patterns"].version,
    title="Tool Use Patterns",
    description="Finds frequent tool sets and sequences (n-grams) and derives association rules to suggest meta-tool opportunities.",
    requires=["tool_calls"],
    params=[
        PluginParam(
            key="min_support",
            type="float",
            title="Min support",
            description="Minimum fraction of traces-with-tools that must contain an itemset for it to be considered frequent.",
            default=0.03,
            min=0.0,
            max=1.0,
        ),
        PluginParam(
            key="max_itemset_size",
            type="int",
            title="Max itemset size",
            description="Maximum size of tool sets considered by Apriori (larger is slower).",
            default=4,
            min=2,
            max=10,
        ),
        PluginParam(
            key="min_confidence",
            type="float",
            title="Min confidence",
            description="Minimum confidence for association rules A→B.",
            default=0.5,
            min=0.0,
            max=1.0,
        ),
        PluginParam(
            key="min_lift",
            type="float",
            title="Min lift",
            description="Minimum lift for association rules A→B (values >1 indicate positive association).",
            default=1.2,
            min=0.0,
            max=10.0,
        ),
        PluginParam(
            key="max_ngram_n",
            type="int",
            title="Max n-gram length",
            description="Maximum length of tool sequences to mine (contiguous n-grams).",
            default=4,
            min=2,
            max=10,
        ),
        PluginParam(
            key="min_ngram_count",
            type="int",
            title="Min n-gram count",
            description="Minimum number of traces an n-gram must appear in to be included.",
            default=5,
            min=1,
            max=1000000,
        ),
        PluginParam(
            key="top_k_itemsets",
            type="int",
            title="Top K itemsets",
            description="How many frequent itemsets to include in the report.",
            default=25,
            min=1,
            max=500,
        ),
        PluginParam(
            key="top_k_rules",
            type="int",
            title="Top K rules",
            description="How many association rules to include in the report.",
            default=25,
            min=1,
            max=500,
        ),
        PluginParam(
            key="top_k_ngrams",
            type="int",
            title="Top K n-grams",
            description="How many frequent sequences to include in the report.",
            default=25,
            min=1,
            max=500,
        ),
    ],
)


def get_plugin(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown plugin '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_plugins() -> List[str]:
    return sorted(_REGISTRY.keys())


def get_plugin_meta(name: str) -> PluginMeta:
    if name not in _META:
        raise KeyError(f"No metadata for plugin '{name}'")
    return _META[name]


def list_plugin_meta() -> List[PluginMeta]:
    return [get_plugin_meta(n) for n in list_plugins()]
