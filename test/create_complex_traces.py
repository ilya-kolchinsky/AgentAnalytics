import os
import time
import uuid
import random
from typing import Dict, Any, Tuple

import mlflow
from mlflow.entities import SpanType


# ----------------------------
# Configuration (env overridable)
# ----------------------------
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "AgentAnalytics_E2E_Banking_Complex")
SUITE_TAG = os.environ.get("AA_SUITE_TAG", "agentanalytics_e2e")

N_TRACES = int(os.environ.get("AA_NUM_TRACES", "300"))  # 200-300
SEED = int(os.environ.get("AA_SEED", "13"))

# Error probabilities (tunable)
P_TRACE_FAIL = float(os.environ.get("AA_P_TRACE_FAIL", "0.06"))     # full trace errors
P_TOOL_FAIL = float(os.environ.get("AA_P_TOOL_FAIL", "0.12"))       # tool span status=ERROR
P_INVALID_ARGS = float(os.environ.get("AA_P_INVALID_ARGS", "0.06")) # missing/empty args to trigger invalid args patterns
P_TIMEOUT = float(os.environ.get("AA_P_TIMEOUT", "0.05"))           # timeout-like tool failures

# Slow down slightly so traces have meaningful timestamp order (helps emerging topics)
SLEEP_BETWEEN = float(os.environ.get("AA_SLEEP", "0.01"))


# ----------------------------
# Synthetic corpora
# ----------------------------
COMMON_INTENTS_EARLY = [
    "balance",
    "nearest_branch",
    "branch_hours",
    "lost_card",
    "reset_password",
]

# Intents that "emerge" later (trend shift detector should pick these up)
EMERGING_INTENTS_LATE = [
    "dispute_charge",
    "international_transfer",
    "exchange_rate",
    "schedule_appointment",
]

# Sentiment variants (user tone)
SENTIMENT_PREFIX = {
    "neutral": ["", "", "", "Hi, ", "Hello, "],
    "positive": ["Thanks! ", "Thank you. ", "Great, can you ", "Quick question: "],
    "negative": ["This is frustrating. ", "I'm angry. ", "This is unacceptable. ", "Why doesn't this work? "],
    "urgent": ["Urgent: ", "ASAP: ", "Please help immediately: "],
}

# Query templates by intent (varied phrasing, incl. some complex prompts)
TEMPLATES = {
    "balance": [
        "What is my current account balance?",
        "Show me my balance for checking.",
        "How much money do I have right now?",
        "Can you give my balance and last 5 transactions?",
        "I need my account balance and a summary of spending categories for the last week.",
        "Please compute my balance excluding pending transactions, and explain how you did it.",
    ],
    "nearest_branch": [
        "What is the address of the branch closest to me?",
        "Find the nearest branch location to my current position.",
        "Where is the closest bank branch?",
        "I’m traveling—find the closest branch and ATM near me and provide directions.",
    ],
    "branch_hours": [
        "Until what time is the branch open today?",
        "What are your opening hours?",
        "Is the branch open on Saturday?",
        "Find branch opening hours for the nearest branch and tell me if it's open right now.",
    ],
    "lost_card": [
        "I lost my credit card. Please block it now.",
        "Cancel my card immediately.",
        "Please freeze my debit card and order a replacement.",
        "My wallet was stolen. Block all cards and tell me next steps. Also list recent suspicious transactions.",
    ],
    "reset_password": [
        "How do I reset my password?",
        "I forgot my password. Help me regain access.",
        "Reset my password and explain the security steps.",
        "I’m locked out. Provide the quickest recovery method without compromising security.",
    ],
    "dispute_charge": [
        "I want to dispute a credit card charge.",
        "There is a fraudulent transaction I don't recognize. Start a dispute.",
        "Help me dispute a charge from 'ACME CORP' on my card and gather the evidence needed.",
        "I need a step-by-step guide to dispute a charge, including timelines and required documents.",
    ],
    "international_transfer": [
        "How do I send an international bank transfer?",
        "Make an international transfer to an IBAN in Germany and estimate fees.",
        "Design the safest procedure for making an international transfer and avoiding scams.",
    ],
    "exchange_rate": [
        "What is today's EUR to USD exchange rate?",
        "Convert 1250 EUR to USD and show the rate used.",
        "Explain how exchange rates are computed and whether your rate includes markup.",
    ],
    "schedule_appointment": [
        "Schedule an appointment with a banker next week.",
        "I need an appointment about a mortgage consultation—find available slots.",
        "Set up an appointment and include required documents in the confirmation.",
    ],
    "change_address": [
        "I moved. How can I change my address on file?",
        "Update my address and ensure my statements go to the new one.",
    ],
    "support_phone": [
        "What is your customer support phone number?",
        "How can I contact a human agent right now?",
    ],
    "transaction_search": [
        "Show my transactions from last month related to groceries.",
        "Find transactions over 200 EUR in the last 30 days and summarize them.",
    ],
}

# Tool catalog (more tools + more sequences)
TOOLS = {
    # identity/auth
    "auth_check": {},
    "auth_refresh": {},
    # account data
    "get_account_balance": {},
    "get_transactions": {},
    "categorize_spend": {},
    # geo/branches
    "geoip_lookup": {},
    "find_nearest_branch": {},
    "get_branch_hours": {},
    "is_branch_open_now": {},
    # card ops
    "block_card": {},
    "freeze_card": {},
    "order_replacement_card": {},
    "flag_suspicious_transactions": {},
    # disputes/transfers
    "search_card_charges": {},
    "open_dispute_case": {},
    "get_dispute_requirements": {},
    "create_international_transfer": {},
    "estimate_transfer_fees": {},
    "get_exchange_rate": {},
    # appointments
    "list_available_appointments": {},
    "schedule_appointment": {},
    # support / profile
    "reset_password_flow": {},
    "change_address": {},
    "get_support_phone": {},
}


# ----------------------------
# Helper: MLflow tool span
# ----------------------------
def tool_span(tool_name: str, args: Dict[str, Any], result: Dict[str, Any], *, ok: bool = True):
    """
    Create a tool span with MLflow inputs/outputs.
    Your TraceView reads tool.name, tool.args, tool.result from attributes or span IO JSON.
    """
    with mlflow.start_span(name=f"tool:{tool_name}", span_type=SpanType.TOOL) as span:
        span.set_inputs({"tool_args": args})
        span.set_attributes({
            "tool.name": tool_name,
            "tool.args": args,
            "tool.status": "OK" if ok else "ERROR",
        })
        span.set_outputs({"tool_result": result})


def maybe_fail_tool(tool_name: str, args: Dict[str, Any], ok_result: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    """
    Return (ok, result, err_signature)
    """
    # Invalid args
    if random.random() < P_INVALID_ARGS:
        return False, {"error": "INVALID_ARGUMENTS", "detail": "Missing required fields"}, "INVALID_ARGUMENTS"

    # Timeout
    if random.random() < P_TIMEOUT:
        return False, {"error": "TIMEOUT", "detail": f"{tool_name} timed out"}, "TIMEOUT"

    # Generic tool fail
    if random.random() < P_TOOL_FAIL:
        code = random.choice(["AUTH_EXPIRED", "RATE_LIMIT", "NOT_FOUND", "UPSTREAM_500"])
        return False, {"error": code, "detail": f"{tool_name} failed with {code}"}, code

    return True, ok_result, ""


# ----------------------------
# Intent routing (with an emerging shift)
# ----------------------------
def sample_intent(i: int, n: int) -> str:
    """
    Early half: mostly common banking intents.
    Late half: keep common ones but increase emerging intents.
    """
    frac = i / max(1, n - 1)
    if frac < 0.5:
        # Heavily weighted to common intents
        weights = {
            "balance": 0.28,
            "nearest_branch": 0.18,
            "branch_hours": 0.14,
            "lost_card": 0.14,
            "reset_password": 0.10,
            "change_address": 0.06,
            "support_phone": 0.05,
            "transaction_search": 0.05,
        }
    else:
        # Late shift: emerging intents gain mass
        weights = {
            "balance": 0.18,
            "nearest_branch": 0.12,
            "branch_hours": 0.10,
            "lost_card": 0.12,
            "reset_password": 0.06,
            "change_address": 0.05,
            "support_phone": 0.04,
            "transaction_search": 0.06,
            "dispute_charge": 0.12,
            "international_transfer": 0.07,
            "exchange_rate": 0.05,
            "schedule_appointment": 0.03,
        }

    r = random.random()
    cum = 0.0
    for k, w in weights.items():
        cum += w
        if r <= cum:
            return k
    return list(weights.keys())[-1]


def sample_sentiment(intent: str) -> str:
    """
    Force some negative sentiment around lost_card/dispute and some positive around balance/branch.
    """
    r = random.random()
    if intent in ("lost_card", "dispute_charge"):
        if r < 0.45:
            return "urgent"
        if r < 0.75:
            return "negative"
        return "neutral"
    if intent in ("balance", "nearest_branch", "branch_hours"):
        if r < 0.25:
            return "positive"
        return "neutral"
    if r < 0.15:
        return "negative"
    return "neutral"


def make_query(intent: str, sentiment: str) -> str:
    base = random.choice(TEMPLATES[intent])
    prefix = random.choice(SENTIMENT_PREFIX[sentiment])
    # Add mild personalization/noise
    if random.random() < 0.20:
        base += " Please."
    if random.random() < 0.10:
        base += " My account is premium."
    if sentiment == "negative" and random.random() < 0.30:
        base += " This is not helpful."
    return prefix + base


# ----------------------------
# Intent execution (tool sequences + occasional trace failures)
# ----------------------------
def execute_intent(intent: str, query: str) -> str:
    """
    Simulate agent tool usage. Sometimes raises exceptions to create trace-level failures.
    Returns final_answer.
    """
    # Auth gate for many intents
    if intent in ("balance", "lost_card", "dispute_charge", "transaction_search", "change_address", "international_transfer"):
        ok, res, sig = maybe_fail_tool("auth_check", {"session_id": "S1"}, {"ok": True})
        tool_span("auth_check", {"session_id": "S1"}, res, ok=ok)
        if not ok:
            # Try refresh once
            ok2, res2, sig2 = maybe_fail_tool("auth_refresh", {"session_id": "S1"}, {"ok": True})
            tool_span("auth_refresh", {"session_id": "S1"}, res2, ok=ok2)
            if not ok2:
                # sometimes fail trace
                if random.random() < 0.6:
                    raise RuntimeError(f"AUTH_FLOW_FAILED::{sig or sig2}")
                return "I couldn’t authenticate you right now. Please try again or log in again."

    if intent == "balance":
        ok, res, sig = maybe_fail_tool("get_account_balance", {"account_id": "A123"}, {"balance_eur": round(200 + random.random()*5000, 2)})
        tool_span("get_account_balance", {"account_id": "A123"}, res, ok=ok)
        if not ok and random.random() < 0.4:
            raise RuntimeError(f"TOOL_FAILED::get_account_balance::{sig}")
        # Optional transactions + categorization path (sequence mining)
        if "last 5 transactions" in query.lower() or "spending categories" in query.lower():
            ok2, res2, sig2 = maybe_fail_tool("get_transactions", {"account_id": "A123", "limit": 5}, {"tx": ["t1", "t2", "t3", "t4", "t5"]})
            tool_span("get_transactions", {"account_id": "A123", "limit": 5}, res2, ok=ok2)
            ok3, res3, sig3 = maybe_fail_tool("categorize_spend", {"tx": res2.get("tx", [])}, {"categories": {"groceries": 120, "transport": 35}})
            tool_span("categorize_spend", {"tx": res2.get("tx", [])}, res3, ok=ok3)
        return "Your current balance is available in your account overview. (Synthetic demo response)"

    if intent == "nearest_branch":
        ok, res, sig = maybe_fail_tool("geoip_lookup", {"ip": "203.0.113.4"}, {"lat": 48.1, "lon": 11.6})
        tool_span("geoip_lookup", {"ip": "203.0.113.4"}, res, ok=ok)
        ok2, res2, sig2 = maybe_fail_tool("find_nearest_branch", {"lat": res.get("lat"), "lon": res.get("lon")}, {"branch_id": "MUC-01", "address": "Example Str. 12, München"})
        tool_span("find_nearest_branch", {"lat": res.get("lat"), "lon": res.get("lon")}, res2, ok=ok2)
        if not ok2 and random.random() < 0.25:
            raise RuntimeError(f"TOOL_FAILED::find_nearest_branch::{sig2}")
        return f"The closest branch is at {res2.get('address','(unknown)')}."

    if intent == "branch_hours":
        # often shares sequence with nearest_branch -> tool patterns should see it
        ok, res, sig = maybe_fail_tool("geoip_lookup", {"ip": "203.0.113.4"}, {"lat": 48.1, "lon": 11.6})
        tool_span("geoip_lookup", {"ip": "203.0.113.4"}, res, ok=ok)
        ok2, res2, sig2 = maybe_fail_tool("find_nearest_branch", {"lat": res.get("lat"), "lon": res.get("lon")}, {"branch_id": "MUC-01", "address": "Example Str. 12, München"})
        tool_span("find_nearest_branch", {"lat": res.get("lat"), "lon": res.get("lon")}, res2, ok=ok2)

        ok3, res3, sig3 = maybe_fail_tool("get_branch_hours", {"branch_id": res2.get("branch_id", "MUC-01")}, {"open": "09:00", "close": "18:00"})
        tool_span("get_branch_hours", {"branch_id": res2.get("branch_id", "MUC-01")}, res3, ok=ok3)
        ok4, res4, sig4 = maybe_fail_tool("is_branch_open_now", {"open": res3.get("open"), "close": res3.get("close")}, {"open_now": random.random() < 0.6})
        tool_span("is_branch_open_now", {"open": res3.get("open"), "close": res3.get("close")}, res4, ok=ok4)

        if not ok3 and random.random() < 0.3:
            raise RuntimeError(f"TOOL_FAILED::get_branch_hours::{sig3}")
        return f"Nearest branch hours are {res3.get('open')}–{res3.get('close')}. Open now: {res4.get('open_now')}."

    if intent == "lost_card":
        # meta-tool opportunity: freeze_card -> order_replacement_card (+ optional flag transactions)
        ok, res, sig = maybe_fail_tool("freeze_card", {"card_id": "C999"}, {"status": "frozen"})
        tool_span("freeze_card", {"card_id": "C999"}, res, ok=ok)
        if not ok and random.random() < 0.5:
            raise RuntimeError(f"TOOL_FAILED::freeze_card::{sig}")

        ok2, res2, sig2 = maybe_fail_tool("order_replacement_card", {"card_id": "C999"}, {"replacement_eta_days": 5})
        tool_span("order_replacement_card", {"card_id": "C999"}, res2, ok=ok2)

        # sometimes also flag suspicious tx (sequence mining)
        if "suspicious" in query.lower() or random.random() < 0.35:
            ok3, res3, sig3 = maybe_fail_tool("flag_suspicious_transactions", {"account_id": "A123"}, {"flagged": ["tx42", "tx77"]})
            tool_span("flag_suspicious_transactions", {"account_id": "A123"}, res3, ok=ok3)

        return "I’ve frozen your card and started a replacement order. (Synthetic demo response)"

    if intent == "reset_password":
        ok, res, sig = maybe_fail_tool("reset_password_flow", {"user_id": "U1"}, {"method": "email_link"})
        tool_span("reset_password_flow", {"user_id": "U1"}, res, ok=ok)
        if not ok and random.random() < 0.2:
            raise RuntimeError(f"TOOL_FAILED::reset_password_flow::{sig}")
        return "To reset your password, we can send a secure recovery link to your email. (Synthetic demo response)"

    if intent == "change_address":
        ok, res, sig = maybe_fail_tool("change_address", {"user_id": "U1", "address": "New St 1"}, {"ok": True})
        tool_span("change_address", {"user_id": "U1", "address": "New St 1"}, res, ok=ok)
        if not ok and random.random() < 0.3:
            raise RuntimeError(f"TOOL_FAILED::change_address::{sig}")
        return "I can help you update your address in your profile settings. (Synthetic demo response)"

    if intent == "support_phone":
        ok, res, sig = maybe_fail_tool("get_support_phone", {}, {"phone": "+49-89-0000-000"})
        tool_span("get_support_phone", {}, res, ok=ok)
        return f"Support phone: {res.get('phone','(unknown)')}"

    if intent == "transaction_search":
        ok, res, sig = maybe_fail_tool("get_transactions", {"account_id": "A123", "range_days": 30}, {"tx": ["tx1", "tx2", "tx3", "tx4"]})
        tool_span("get_transactions", {"account_id": "A123", "range_days": 30}, res, ok=ok)
        if not ok and random.random() < 0.3:
            raise RuntimeError(f"TOOL_FAILED::get_transactions::{sig}")
        return "Here are the transactions matching your filters. (Synthetic demo response)"

    if intent == "dispute_charge":
        # Frequent chain: search_card_charges -> open_dispute_case -> get_dispute_requirements
        ok, res, sig = maybe_fail_tool("search_card_charges", {"query": "ACME CORP"}, {"charges": ["chg1", "chg2"]})
        tool_span("search_card_charges", {"query": "ACME CORP"}, res, ok=ok)
        ok2, res2, sig2 = maybe_fail_tool("open_dispute_case", {"charge_id": "chg1"}, {"case_id": "D-1001"})
        tool_span("open_dispute_case", {"charge_id": "chg1"}, res2, ok=ok2)
        ok3, res3, sig3 = maybe_fail_tool("get_dispute_requirements", {"case_type": "fraud"}, {"requirements": ["statement", "police_report_optional"]})
        tool_span("get_dispute_requirements", {"case_type": "fraud"}, res3, ok=ok3)

        # Make this one error-prone to feed failure taxonomy
        if (not ok2 or not ok3) and random.random() < 0.5:
            raise RuntimeError(f"TOOL_FAILED::dispute_flow::{sig2 or sig3}")

        return "I started a dispute case and can guide you through next steps. (Synthetic demo response)"

    if intent == "international_transfer":
        ok, res, sig = maybe_fail_tool("estimate_transfer_fees", {"amount_eur": 500, "dest": "DE"}, {"fee_eur": 6.5})
        tool_span("estimate_transfer_fees", {"amount_eur": 500, "dest": "DE"}, res, ok=ok)
        ok2, res2, sig2 = maybe_fail_tool("create_international_transfer", {"amount_eur": 500, "iban": "DE123..."}, {"transfer_id": "T-77"})
        tool_span("create_international_transfer", {"amount_eur": 500, "iban": "DE123..."}, res2, ok=ok2)
        if not ok2 and random.random() < 0.35:
            raise RuntimeError(f"TOOL_FAILED::create_international_transfer::{sig2}")
        return "International transfer initiated with estimated fees. (Synthetic demo response)"

    if intent == "exchange_rate":
        ok, res, sig = maybe_fail_tool("get_exchange_rate", {"pair": "EURUSD"}, {"rate": round(1.05 + random.random()*0.08, 4)})
        tool_span("get_exchange_rate", {"pair": "EURUSD"}, res, ok=ok)
        return f"EUR→USD rate is {res.get('rate')} (Synthetic demo response)."

    if intent == "schedule_appointment":
        ok, res, sig = maybe_fail_tool("list_available_appointments", {"topic": "mortgage"}, {"slots": ["2026-03-01 10:00", "2026-03-02 14:00"]})
        tool_span("list_available_appointments", {"topic": "mortgage"}, res, ok=ok)
        ok2, res2, sig2 = maybe_fail_tool("schedule_appointment", {"slot": "2026-03-01 10:00"}, {"confirmation_id": "APT-9"})
        tool_span("schedule_appointment", {"slot": "2026-03-01 10:00"}, res2, ok=ok2)
        if not ok2 and random.random() < 0.3:
            raise RuntimeError(f"TOOL_FAILED::schedule_appointment::{sig2}")
        return "Appointment scheduled. (Synthetic demo response)"

    return "I can help with that. (Synthetic demo response)"


# ----------------------------
# One trace end-to-end
# ----------------------------
def one_trace(query: str, run_id: str, user_id: str, intent: str):
    with mlflow.start_span(name="agent_request", span_type=SpanType.CHAIN) as root:
        # Trace inputs/outputs (MLflow stores as mlflow.traceInputs/Outputs)
        root.set_inputs({"query_text": query, "intent": intent})

        mlflow.update_current_trace(
            tags={
                "aa.suite": SUITE_TAG,
                "aa.run": run_id,
                "aa.kind": "synthetic",
                "aa.scenario": "banking_demo_v2",
                "aa.intent": intent,
            },
            metadata={
                "user_id": user_id,
                "generator": "tools/generate_bank_traces_v2.py",
            },
        )

        # Sometimes create a pure trace-level failure (no tools), to diversify failure taxonomy
        if random.random() < P_TRACE_FAIL:
            root.set_outputs({"final_answer": "I encountered an internal error. Please try again."})
            raise RuntimeError("TRACE_FAILURE::INTERNAL_AGENT_ERROR")

        answer = execute_intent(intent, query)
        root.set_outputs({"final_answer": answer})


def main():
    random.seed(SEED)

    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception:
        pass

    run_id = uuid.uuid4().hex[:10]
    print("Generating banking synthetic traces (v2)")
    print("Tracking URI:", TRACKING_URI)
    print("Experiment:", EXPERIMENT_NAME)
    print("Suite tag:", SUITE_TAG)
    print("Run tag:", run_id)
    print("Count:", N_TRACES)

    # Associate with an MLflow run (optional; ok on shared server)
    with mlflow.start_run(run_name=f"AgentAnalytics_BankingV2_{run_id}"):
        for i in range(N_TRACES):
            intent = sample_intent(i, N_TRACES)
            sentiment = sample_sentiment(intent)
            query = make_query(intent, sentiment)
            user = f"user_{random.randint(1, 30)}"

            try:
                one_trace(query, run_id=run_id, user_id=user, intent=intent)
            except Exception:
                # Exceptions are expected to create ERROR traces/spans and feed your RCA plugin
                pass

            time.sleep(SLEEP_BETWEEN)

    print("\nDone. Use these filters to isolate:")
    print(f"tag.aa.suite = '{SUITE_TAG}' AND tag.aa.run = '{run_id}'")
    print("Example MLflow trace query (put into your YAML mlflow.query):")
    print(f"tag.aa.scenario = 'banking_demo_v2' AND tag.aa.run = '{run_id}'")


if __name__ == "__main__":
    main()
