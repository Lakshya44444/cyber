"""Loaders for the external, India-specific test sets (never used for training
the base model).

  iiitd_2011       IIIT-Delhi crowdsourced Indian SMS (Yadav, Kumaraguru et al., 2011):
                   1,000 ham (much of it Hinglish chat) + 1,000 promotional spam.
  india_telecom    Indian Telecom SMS Spam Collection (junioralive, 2024, MIT):
                   1,522 ham + 745 promotional / OTP "spam".
  probes           Hand-written Indian scam probes and legitimate look-alikes from the
                   spam-scan project (Rishabh Aryan, 2026, MIT): UPI, KYC, courier,
                   digital arrest, job, investment ... in English, Hinglish, Hindi and
                   misspelled form. Parsed with `ast` (the file is never executed).

Promotional spam is NOT a scam. For ScamTrail it is a benign message, so it
counts toward the false-alert rate.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd

from data import normalise_for_template

EXT = Path("data/external")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
HINGLISH_WORDS = re.compile(
    r"\b(aap|aapka|aapke|aapko|apna|apne|karein|karo|kare|hai|hain|ho|ke liye|turant|warna|bhai|"
    r"nahi|kar|paise|daalein|dalein|jayega|gaya|mein|yaar|abhi|jaldi|kya|sirf|roz|ghar|baithe|"
    r"bhej|diye|priy|grahak|mila|jeete|inaam|bijli)\b", re.I)


def lang_of(text: str, hint: str = "") -> str:
    if DEVANAGARI.search(text):
        return "Hindi"
    if hint:
        return hint
    return "Hinglish" if len(HINGLISH_WORDS.findall(text)) >= 2 else "English"


def _pairs(node):
    """Yield (label_name, text) from list / dict literals of (SPAM|HAM, "text") tuples."""
    for t in ast.walk(node):
        if isinstance(t, ast.Tuple) and len(t.elts) == 2 and isinstance(t.elts[0], ast.Name) \
                and isinstance(t.elts[1], ast.Constant) and isinstance(t.elts[1].value, str):
            yield t.elts[0].id, t.elts[1].value


def load_probes() -> pd.DataFrame:
    tree = ast.parse((EXT / "spamscan_probes.py.txt").read_text(encoding="utf-8"))
    rows = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            if name not in {"HINGLISH", "INDIA_SCAMS", "MISSPELLED"}:
                continue
            if name == "INDIA_SCAMS" and isinstance(n.value, ast.Dict):
                for k, v in zip(n.value.keys, n.value.values):
                    for lab, txt in _pairs(v):
                        rows.append((lab, txt, k.value, lang_of(txt)))
            else:
                hint = "Misspelled" if name == "MISSPELLED" else ""
                for lab, txt in _pairs(n.value):
                    rows.append((lab, txt, name.lower(), lang_of(txt, hint)))
    ex = json.loads((EXT / "examples.json").read_text(encoding="utf-8"))
    for c in ex["categories"]:
        for e in c["examples"]:
            if e["expected"] in ("spam", "ham"):
                rows.append(("SPAM" if e["expected"] == "spam" else "HAM", e["text"], c["id"],
                             lang_of(e["text"], e["lang"] if e["lang"] != "English" else "")))
    df = pd.DataFrame(rows, columns=["lab", "text", "category", "lang"])
    df["is_scam"] = (df["lab"] == "SPAM").astype(int)
    df = df.drop_duplicates("text").reset_index(drop=True)
    return df[["text", "is_scam", "category", "lang"]]


def load_benign_indian() -> pd.DataFrame:
    """Indian SMS traffic that is NOT a scam: ham and promotional spam."""
    a = pd.read_csv(EXT / "iiitd_2011.csv").rename(columns={"v1": "label", "v2": "text"})
    a["source"] = "IIIT-D 2011"
    b = pd.read_csv(EXT / "india_telecom_2024.csv").rename(columns={"Msg": "text", "Label": "label"})
    b["source"] = "Indian Telecom 2024"
    d = pd.concat([a, b], ignore_index=True)
    d["text"] = d["text"].astype(str).str.strip()
    d = d[d["text"].str.len() > 3].drop_duplicates("text").reset_index(drop=True)
    d["kind"] = d["label"].str.lower().map({"ham": "ham", "spam": "promo"})
    d["lang"] = d["text"].map(lang_of)
    d["group"] = d["text"].map(normalise_for_template).factorize()[0]
    return d[["text", "kind", "source", "lang", "group"]]


if __name__ == "__main__":
    p = load_probes()
    print(p.groupby(["lang", "is_scam"]).size().unstack(fill_value=0))
    print(p["category"].value_counts().to_dict())
    b = load_benign_indian()
    print(b.groupby(["source", "kind"]).size(), b["lang"].value_counts().to_dict())
