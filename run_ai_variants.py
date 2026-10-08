"""AI-enabled scam stress test.

AI lets scammers write fluent, polite, personalised, code-mixed messages with
none of the classic red-flag words. Question: does a single-message detector
still catch them, and does the journey-level decision still hold?

  1. Message level: recall at the same 2% budget on (a) the classic Indian
     probes and (b) 50 LLM-written variants, for the public-data model and the
     India-template model.
  2. Journey level: rerun the journey benchmark with every scam message
     replaced by an AI variant (env AI_TEXTS=1) and compare.
Writes results/ai_variants.json.
"""
import json
import os
import subprocess
import sys

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from conformal import conformal_threshold
from data import load_smishing
from external import load_benign_indian, load_probes
from message_model import MessageModel
from synth_india import generate

AI = json.load(open("data/ai_variants.json"))
df, probes, benign = load_smishing(), load_probes(), load_benign_indian()
R = {"n_ai_scam": len(AI["scam"]), "n_ai_legit": len(AI["legit_lookalike"]), "message_level": {}}
runs = {"public_only": [], "india_templates": []}
for seed in range(10):
    tr, cal = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(df, df.is_smish, df.group))
    dtr, dcal = df.iloc[tr], df.iloc[cal]
    ad, _ = next(GroupShuffleSplit(1, test_size=0.5, random_state=seed).split(benign, groups=benign.group))
    badapt = benign.iloc[ad]
    a_tr, a_cal = next(GroupShuffleSplit(1, test_size=0.5, random_state=seed + 100).split(badapt, groups=badapt.group))
    b_tr, b_cal = badapt.iloc[a_tr], badapt.iloc[a_cal]
    syn = generate(seed=seed)
    for name, (X, y) in {
        "public_only": (dtr.text.values, dtr.is_smish.values),
        "india_templates": (np.concatenate([dtr.text.values, b_tr.text.values, syn.text.values]),
                            np.concatenate([dtr.is_smish.values, np.zeros(len(b_tr), int), syn.is_smish.values])),
    }.items():
        m = MessageModel("logreg").fit(X, y)
        thr = conformal_threshold(m.score(dcal.text.values[dcal.is_smish.values == 0]), 0.02)
        if name == "india_templates":
            thr = max(thr, conformal_threshold(m.score(b_cal.text.values), 0.02))
        runs[name].append({
            "classic_indian_probes": float((m.score(probes.text[probes.is_scam == 1].values) > thr).mean()),
            "ai_written_scams": float((m.score(AI["scam"]) > thr).mean()),
            "ai_legit_lookalikes_flagged": float((m.score(AI["legit_lookalike"]) > thr).mean()),
        })
for k, v in runs.items():
    R["message_level"][k] = {kk: float(np.mean([d[kk] for d in v])) for kk in v[0]}
print(json.dumps(R["message_level"], indent=1))

# journey level: same benchmark, scam texts replaced by AI variants
env = dict(os.environ, AI_TEXTS="1", OUT="results/journey_ai.json")
subprocess.run([sys.executable, "run_journey.py"], env=env, check=True, stdout=subprocess.DEVNULL)
JA, J = json.load(open("results/journey_ai.json")), json.load(open("results/journey.json"))
pick = lambda d: {"T2_recall": d["decision"]["T2_or_higher_prompt"]["scam_payment_recall"],
                  "T2_false_alert": d["decision"]["T2_or_higher_prompt"]["false_alert_rate_genuine"],
                  "T3_recall": d["decision"]["T3_hold"]["scam_payment_recall"],
                  "single_sms_recall": d["methods"]["Single-SMS classifier"]["scam_payment_recall"]}
R["journey_level"] = {"classic_scam_texts": pick(J), "ai_written_scam_texts": pick(JA)}
json.dump(R, open("results/ai_variants.json", "w"), indent=2)
print(json.dumps(R["journey_level"], indent=1))
