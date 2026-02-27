import re
from typing import Optional

_WS = re.compile(r"\s+")


def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = _WS.sub(" ", s)
    return s


def norm_numbers(s: str) -> str:
    s = norm_text(s)
    # Replace email-ish, UUID-ish, long ids, numbers
    s = re.sub(r"\b[\w.%-]+@[\w.-]+\.[a-z]{2,}\b", "<email>", s)
    s = re.sub(r"\b[0-9a-f]{8,}\b", "<hex>", s)
    s = re.sub(r"\b\d{4,}\b", "<num>", s)
    s = re.sub(r"\b\d+\b", "<num>", s)
    s = _WS.sub(" ", s).strip()
    return s


def shorten(s: Optional[str], max_len: int = 240) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."
