import os
import time
import uuid
import random
from typing import Dict, Any

import mlflow
from mlflow.entities import SpanType

# ---- configuration ----
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "AgentAnalytics_E2E_Banking_Simple")
SUITE_TAG = os.environ.get("AA_SUITE_TAG", "agentanalytics_e2e")


FAQ_CLUSTERS = [
    # cluster: account balance
    [
        "What is my current account balance?",
        "How much money do I have in my account?",
        "Show my balance please",
        "What’s the balance on my checking account?"
    ],
    # cluster: closest branch
    [
        "What is the address of the branch closest to me?",
        "Find the nearest branch location",
        "Where is the closest bank branch?",
        "Closest branch address near my location"
    ],
    # cluster: branch opening hours
    [
        "Until what time is the branch open today?",
        "What are your opening hours?",
        "Is the branch open on Saturday?",
        "When does the branch close?"
    ],
    # cluster: card cancellation
    [
        "Cancel my card immediately",
        "I lost my card, how do I block it?",
        "Please freeze my debit card",
        "How can I disable my card?"
    ],
    # cluster: simple FAQs
    [
        "What is your customer support phone number?",
        "How do I reset my password?",
        "Where can I change my address?"
    ],
]


def fake_tool_call(tool_name: str, args: Dict[str, Any], result: Dict[str, Any], ok: bool = True):
    # Create a span that your analytics can interpret as a tool call:
    with mlflow.start_span(name=f"tool:{tool_name}", span_type=SpanType.TOOL) as span:
        span.set_inputs({"tool_args": args})
        span.set_attributes({
            "tool.name": tool_name,
            "tool.args": args,
            "tool.status": "OK" if ok else "ERROR",
        })
        span.set_outputs({"tool_result": result})


def one_trace(query: str, run_id: str, user_id: str):
    # Trace root span (CHAIN) representing the agent request
    with mlflow.start_span(name="agent_request", span_type=SpanType.CHAIN) as root:
        # Put query in inputs where your TraceView can find it
        root.set_inputs({"query_text": query})

        # Add trace-level context tags/metadata (critical for isolation)
        mlflow.update_current_trace(
            tags={
                "aa.suite": SUITE_TAG,
                "aa.run": run_id,
                "aa.kind": "synthetic",
            },
            metadata={
                "user_id": user_id,
                "generator": "tools/generate_test_traces.py",
            },
        )

        # Simulate a plausible tool path + final answer
        ql = query.lower()
        if "balance" in ql:
            fake_tool_call("get_account_balance", {"account_id": "A123"}, {"balance_eur": 1234.56})
            answer = "Your current balance is €1,234.56."
        elif "closest" in ql or "nearest" in ql or "branch" in ql and "address" in ql:
            fake_tool_call("geoip_lookup", {"ip": "203.0.113.4"}, {"lat": 48.1, "lon": 11.6})
            fake_tool_call("find_nearest_branch", {"lat": 48.1, "lon": 11.6}, {"address": "Example Str. 12, München"})
            answer = "The closest branch is at Example Str. 12, München."
        elif "open" in ql or "close" in ql or "hours" in ql:
            fake_tool_call("get_branch_hours", {"branch_id": "MUC-01"}, {"close_time": "18:00", "open_time": "09:00"})
            answer = "Today the branch is open 09:00–18:00."
        elif "lost" in ql or "cancel" in ql or "block" in ql or "freeze" in ql:
            # Sometimes fail to give you RCA/error signal diversity
            if random.random() < 0.2:
                fake_tool_call("block_card", {"card_id": "C999"}, {"error": "AUTH_EXPIRED"}, ok=False)
                # simulate trace error
                root.set_outputs({"final_answer": "I couldn’t block the card due to an authentication issue."})
                raise RuntimeError("AUTH_EXPIRED while calling block_card")
            else:
                fake_tool_call("block_card", {"card_id": "C999"}, {"status": "blocked"})
                answer = "I have blocked your card. If you find it later, you can unblock it in the app."
        else:
            # simple FAQ path
            answer = "You can do that in the app under Settings → Account. If you need help, contact support."

        root.set_outputs({"final_answer": answer})


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_id = uuid.uuid4().hex[:10]
    print("Generating synthetic traces")
    print("Tracking URI:", TRACKING_URI)
    print("Suite tag:", SUITE_TAG)
    print("Run tag:", run_id)

    # You can also associate traces with a run (optional)
    with mlflow.start_run(run_name=f"AgentAnalytics_E2E_{run_id}"):
        for i in range(120):
            cluster = random.choice(FAQ_CLUSTERS)
            q = random.choice(cluster)
            user = f"user_{random.randint(1, 20)}"
            try:
                one_trace(q, run_id=run_id, user_id=user)
            except Exception:
                # exceptions are fine; they create ERROR traces/spans
                pass
            time.sleep(0.02)

    print("Done. Use these tags to filter:")
    print(f"tag.aa.suite = '{SUITE_TAG}' AND tag.aa.run = '{run_id}'")


if __name__ == "__main__":
    main()
