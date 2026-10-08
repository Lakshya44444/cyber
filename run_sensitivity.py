"""Sensitivity sweep: is the ranking of methods an artefact of our simulator
settings? Re-runs the journey benchmark with weaker scam signals, fewer shared
messages and more hard negatives, and records every method at the same 2%
false-alert budget. Writes results/sensitivity.json."""
import itertools
import json
import os
import subprocess
import sys

GRID = list(itertools.product([1.0, 0.7, 0.5], [1.0, 0.5, 0.0], [1.0, 2.0]))  # signal, share, hard
out = []
os.makedirs("results/sweep", exist_ok=True)
for sig, sh, hd in GRID:
    f = f"results/sweep/j_{sig}_{sh}_{hd}.json"
    env = dict(os.environ, SIGNAL=str(sig), SHARE=str(sh), HARD=str(hd), OUT=f)
    subprocess.run([sys.executable, "run_journey.py"], env=env, check=True, stdout=subprocess.DEVNULL)
    J = json.load(open(f))
    row = {"signal": sig, "share": sh, "hard": hd,
           "recall": {k: v["scam_payment_recall"] for k, v in J["methods"].items()},
           "T2_prompt_recall": J["decision"]["T2_or_higher_prompt"]["scam_payment_recall"],
           "false_alert": {k: v["false_alert_rate_genuine"] for k, v in J["methods"].items()},
           "hard_neg_false_alert": {k: v["false_alert_rate_hard_negatives"] for k, v in J["methods"].items()},
           "T3_hold": {k: J["decision"]["T3_hold"][k] for k in
                       ["scam_payment_recall", "false_alert_rate_genuine", "scam_journeys_flagged"]}}
    out.append(row)
    print(sig, sh, hd, {k[:12]: round(v, 3) for k, v in row["recall"].items()}, flush=True)
ml = ["Single-SMS classifier", "Payment-only model", "ScamTrail without journey state", "ScamTrail (full)"]
summary = {
    "settings": len(out),
    "full_is_best_learned_method": sum(max(ml, key=lambda m: r["recall"][m]) == "ScamTrail (full)" for r in out),
    "full_beats_payment_only_by_pts": {"min": min((r["recall"]["ScamTrail (full)"] - r["recall"]["Payment-only model"]) * 100 for r in out),
                                       "max": max((r["recall"]["ScamTrail (full)"] - r["recall"]["Payment-only model"]) * 100 for r in out)},
    "full_recall_range": [min(r["recall"]["ScamTrail (full)"] for r in out), max(r["recall"]["ScamTrail (full)"] for r in out)],
    "payment_only_recall_range": [min(r["recall"]["Payment-only model"] for r in out), max(r["recall"]["Payment-only model"] for r in out)],
}
json.dump({"grid": out, "summary": summary}, open("results/sensitivity.json", "w"), indent=2)
print(json.dumps(summary, indent=1))
