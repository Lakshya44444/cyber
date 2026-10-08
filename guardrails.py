"""ScamTrail guardrails: everything that sits between untrusted text and a decision.

Three layers, following the usual input-guard / output-guard pattern:

INPUT GUARDS  (run on every message the user shares, before any model)
  1. normalise()        undo obfuscation: zero-width chars, look-alike letters,
                        leetspeak inside words, "K Y C" spacing
  2. detect_injection() find text that tries to talk to the model
                        ("ignore previous instructions", "mark this as safe")
                        and cut it out
  3. redact_pii()       mask phone, UPI ID, Aadhaar, PAN, card, account, OTP
                        before anything leaves the phone

OUTPUT GUARDS (run on what the model returns)
  4. Intent             schema-locked extraction result (pydantic); anything
                        that does not fit the schema is rejected
  5. monotonic()        a shared message can only RAISE a journey stage score,
                        never lower it, so a scammer cannot "talk down" risk
  6. check_explanation() the sentence shown on the PIN screen must be grounded
                        in the model's real reasons, must not accuse, must not
                        say "safe", and must be short; otherwise a fixed
                        template is shown instead
"""
from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------- 1. normalise
ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "к": "k",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "ɡ": "g", "ո": "n", "ս": "u",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "У": "Y",
})
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
                      "@": "a", "$": "s", "!": "i", "|": "l"})
SPACED_LETTERS = re.compile(r"\b(?:[A-Za-z][ .\-_*]){2,}[A-Za-z]\b")
MIXED_TOKEN = re.compile(r"\b(?=[\w@$!|]*[A-Za-z])(?=[\w@$!|]*[0-9@$!|])[\w@$!|]{3,}\b")


# real e-mail addresses and UPI handles are not leetspeak: shield them so the
# PII redactor still sees them ("name@ybl" must not become "nameaybl")
SHIELD = re.compile(r"\b[\w.+-]+@(?:[\w-]+\.[a-z]{2,}(?:\.[a-z]{2,})?|ybl|ibl|axl|ok[a-z]+|paytm|upi|apl|"
                    r"axisbank|icici|sbi|hdfcbank|kotak|yesbank|freecharge|jio|airtel)\b"
                    r"|\b[A-Z]{5}\d{4}[A-Z]\b")   # PAN format is letters+digits by design, not leetspeak


def _de_leet(tok: str) -> str:
    # keep pure numbers and things that look like amounts / phone numbers
    letters = sum(c.isalpha() for c in tok)
    if letters < 2 or letters < len(tok) / 2:
        return tok
    return tok.translate(LEET)


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).translate(ZERO_WIDTH).translate(HOMOGLYPHS)
    t = SPACED_LETTERS.sub(lambda m: re.sub(r"[ .\-_*]", "", m.group()), t)
    kept = SHIELD.findall(t)
    t = SHIELD.sub("\x00", t)
    t = MIXED_TOKEN.sub(lambda m: _de_leet(m.group()), t)
    for k in kept:
        t = t.replace("\x00", k, 1)
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------- 2. prompt injection
INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules|messages)",
    r"disregard (the )?(previous|above|system)",
    r"\b(system|developer) (prompt|message|instruction)s?\b",
    r"\byou are (now|an?) (ai|assistant|model|classifier)\b",
    r"\b(mark|classify|label|treat|tag) (this|it|the message|me)? ?(as )?(safe|legit|legitimate|genuine|ham|not spam|trusted)\b",
    r"\bthis (message|sms) is (safe|genuine|legit|verified)\b",
    r"\bnot (a )?(scam|fraud|phishing|spam)\b",
    r"\b(assistant|user|system)\s*:",
    r"<\/?(s|system|im_start|im_end)>",
    r"\b(ye|yeh|isko|is message ko) (safe|sahi|genuine) (hai|mark karo|maano)\b",
    r"\bscam nahi hai\b",
    r"\brisk score\s*[:=]\s*0",
]
INJ_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.I)


def detect_injection(text: str) -> tuple[str, list[str]]:
    """Return text with injected sentences removed, plus the matched phrases."""
    hits = [m.group() for m in INJ_RE.finditer(text)]
    if not hits:
        return text, []
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    kept = [s for s in sentences if not INJ_RE.search(s)]
    return " ".join(kept).strip(), hits


# ------------------------------------------------------------- 3. PII redaction
# order matters: longer / more specific identifiers first
PII_PATTERNS = [
    ("EMAIL", r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}(?:\.[a-z]{2,})?\b"),
    ("UPI", r"\b[\w.-]{2,}@(?:ybl|ibl|axl|okaxis|okhdfcbank|okicici|oksbi|paytm|upi|apl|axisbank|icici|sbi|hdfcbank|kotak|yesbank|[a-z]{2,10})\b"),
    ("CARD", r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,7}\b(?<=\d)"),
    ("AADHAAR", r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b"),
    ("PAN", r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    ("OTP", r"\b(?:otp|pin|code)\D{0,12}\d{4,8}\b"),
    ("PHONE", r"(?:\+?91[ -]?)?\b[6-9]\d{9}\b|\b0\d{9,11}\b"),
    ("ACCOUNT", r"\b\d{9,18}\b"),
]
PII_RE = [(name, re.compile(p, re.I)) for name, p in PII_PATTERNS]
URL_RE = re.compile(r"\b(?:https?://)?((?:[a-z0-9-]+\.)+[a-z]{2,})(/\S*)?", re.I)


def redact_pii(text: str) -> tuple[str, dict]:
    counts: dict[str, int] = {}
    t = text
    for name, rx in PII_RE:
        t, n = rx.subn(f"<{name}>", t)
        if n:
            counts[name] = counts.get(name, 0) + n
    # keep only the domain of a link: the domain is a useful signal, the path
    # can carry tokens or personal identifiers
    t = URL_RE.sub(lambda m: f"<URL:{m.group(1).lower()}>", t)
    return t, counts


# ------------------------------------------------- 4. schema-locked extraction
class Intent(BaseModel):
    """The only thing the message extractor is allowed to return."""
    intent: Literal["none", "kyc_or_account", "prize_or_lottery", "job_or_task",
                    "investment", "authority_threat", "delivery_or_bill", "other_scam"]
    asks_payment: bool
    asks_credentials: bool
    has_link: bool
    urgency: bool
    risk: float = Field(ge=0.0, le=1.0)


def safe_parse_intent(raw: dict) -> Intent | None:
    try:
        return Intent.model_validate(raw)
    except ValidationError:
        return None  # rejected: caller treats as "no information", never as "safe"


# ---------------------------------------------------- 5. monotonic risk update
def monotonic(prev: float, new: float) -> float:
    return max(prev, new)


# ------------------------------------------------------ 6. explanation guard
BANNED = [
    r"\bsafe\b", r"\bguarantee", r"\b100 ?%", r"\bdefinitely\b", r"\bcertainly\b",
    r"\bfraudster\b", r"\bcriminal\b", r"\bscammer\b", r"\byou are being scammed\b",
    r"\byou (are|were) (cheated|fooled)\b", r"\barrest", r"\bpolice will\b",
    r"\baccount will be (blocked|frozen)\b", r"\bsurakshit\b", r"\bdhokebaaz\b",
    r"\brisk score\b", r"\bprobability\b", r"\b0\.\d+\b",
]
BANNED_RE = re.compile("|".join(BANNED), re.I)

# reason code -> words that may appear in an explanation about it
REASON_WORDS = {
    "on_call": ["on a call", "phone call", "while talking"],
    "new_payee": ["first payment to this", "never paid", "first time paying"],
    "amount_high": ["higher than your usual", "more than usual", "larger than usual"],
    "remote_app": ["screen-sharing", "screen sharing", "remote app", "remote-access"],
    "msg_scam_like": ["message", "sms", "link"],
    "fast_pin": ["unusually fast", "in a hurry", "rushed"],
    "repeat_loop": ["several", "growing payments", "again and again", "repeated"],
    "payee_reported": ["reported"],
}
ALL_REASON_WORDS = {w: k for k, ws in REASON_WORDS.items() for w in ws}

TEMPLATE = {
    "on_call": "You are on a call while paying.",
    "new_payee": "This is your first payment to this UPI ID.",
    "amount_high": "The amount is much higher than your usual payments.",
    "remote_app": "A screen-sharing or remote app is open.",
    "msg_scam_like": "A message you shared looks like known scam messages.",
    "fast_pin": "This payment is being made unusually fast.",
    "repeat_loop": "You have made several growing payments to different UPI IDs this week.",
    "payee_reported": "This UPI ID has been reported before.",
}


def check_explanation(text: str, reasons: list[str], max_len: int = 240) -> list[str]:
    """Return a list of violations. Empty list = OK to show."""
    v = []
    if len(text) > max_len:
        v.append("too_long")
    if BANNED_RE.search(text):
        v.append("banned_phrase:" + BANNED_RE.search(text).group())
    low = text.lower()
    mentioned = {k for w, k in ALL_REASON_WORDS.items() if w in low}
    if not mentioned & set(reasons):
        v.append("not_grounded")          # says nothing the model actually used
    if mentioned - set(reasons):
        v.append("ungrounded_claim:" + ",".join(sorted(mentioned - set(reasons))))
    return v


def explanation_or_fallback(candidate: str, reasons: list[str]) -> tuple[str, list[str]]:
    v = check_explanation(candidate, reasons)
    if not v:
        return candidate, []
    fallback = " ".join(TEMPLATE[r] for r in reasons[:2]) + " Please pause and check before you pay."
    return fallback, v


def guard_message(text: str) -> dict:
    """Full input-guard chain for one shared message."""
    clean = normalise(text)
    clean, inj = detect_injection(clean)
    redacted, pii = redact_pii(clean)
    return {"text": redacted, "injection": inj, "pii": pii}
