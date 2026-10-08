"""External validation on India-specific data the model never trained on.

Question 1 (transfer): a model trained only on the Mishra & Soni SMS dataset,
with its 2% alert budget set on that dataset - how does it do on Indian
Hinglish / Hindi scams and on everyday Indian SMS traffic?

Question 2 (adaptation = what shadow mode does): if a bank collects a few
weeks of its own NON-scam traffic (no scam labels needed) and
  (b) only re-sets the conformal threshold on it, or
  (c) also adds it to training as negatives,
does the false-alert budget come back, and is scam recall kept?

Writes results/external.json.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from conformal import conformal_threshold
from data import load_smishing
from external import load_benign_indian, load_probes
from message_model import MessageModel
from synth_india import generate

ALPHA = 0.02
SEEDS = range(10)
df = load_smishing()
probes = load_probes()
benign = load_benign_indian()
R = {"sizes": {"mishra_train_pool": len(df), "indian_benign": int(len(benign)),
               "indian_benign_by_source_kind": {f"{s}|{k}": int(n) for (s, k), n in
                                                benign.groupby(["source", "kind"]).size().items()},
               "probes_scam": int(probes["is_scam"].sum()),
               "probes_legit_lookalike": int((probes["is_scam"] == 0).sum()),
               "probes_by_lang": {f"{l}|{'scam' if s else 'legit'}": int(n) for (l, s), n in
                                  probes.groupby(["lang", "is_scam"]).size().items()}},
     "alpha": ALPHA}


def report(score_probe, score_benign_test, thr, btest):
    ps = probes.assign(a=score_probe > thr)
    out = {
        "scam_recall_all": float(ps.loc[ps.is_scam == 1, "a"].mean()),
        "scam_recall_by_lang": ps[ps.is_scam == 1].groupby("lang")["a"].mean().to_dict(),
        "lookalike_false_alert": float(ps.loc[ps.is_scam == 0, "a"].mean()),
        "indian_ham_false_alert": float((score_benign_test > thr)[btest["kind"].values == "ham"].mean()),
        "indian_promo_flag_rate": float((score_benign_test > thr)[btest["kind"].values == "promo"].mean()),
    }
    out["indian_benign_false_alert"] = float((score_benign_test > thr).mean())
    out["hinglish_ham_false_alert"] = float((score_benign_test > thr)[
        (btest["lang"].values == "Hinglish") & (btest["kind"].values == "ham")].mean())
    return out


runs = {"a_base": [], "b_recalibrate": [], "c_retrain_recalibrate": [], "d_plus_synthetic_india": []}
for seed in SEEDS:
    tr, cal = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(df, df.is_smish, df.group))
    dtr, dcal = df.iloc[tr], df.iloc[cal]
    ad, te = next(GroupShuffleSplit(1, test_size=0.5, random_state=seed).split(benign, groups=benign.group))
    badapt, btest = benign.iloc[ad], benign.iloc[te]
    # split the adaptation traffic: half for training negatives, half for calibration
    a_tr, a_cal = next(GroupShuffleSplit(1, test_size=0.5, random_state=seed + 100)
                       .split(badapt, groups=badapt.group))
    b_tr, b_cal = badapt.iloc[a_tr], badapt.iloc[a_cal]

    # (a) trained and calibrated on Mishra only
    m = MessageModel("logreg").fit(dtr.text.values, dtr.is_smish.values)
    thr_a = conformal_threshold(m.score(dcal.text.values[dcal.is_smish.values == 0]), ALPHA)
    sp, sb = m.score(probes.text.values), m.score(btest.text.values)
    runs["a_base"].append(report(sp, sb, thr_a, btest))
    # (b) same model, threshold re-set on Indian benign traffic (no scam labels used)
    neg_b = np.concatenate([m.score(dcal.text.values[dcal.is_smish.values == 0]), m.score(b_cal.text.values)])
    thr_b = max(thr_a, conformal_threshold(m.score(b_cal.text.values), ALPHA))
    runs["b_recalibrate"].append(report(sp, sb, thr_b, btest))
    # (c) Indian benign traffic also added as training negatives, then recalibrated
    X = np.concatenate([dtr.text.values, b_tr.text.values])
    y = np.concatenate([dtr.is_smish.values, np.zeros(len(b_tr), int)])
    m2 = MessageModel("logreg").fit(X, y)
    thr_c = max(conformal_threshold(m2.score(dcal.text.values[dcal.is_smish.values == 0]), ALPHA),
                conformal_threshold(m2.score(b_cal.text.values), ALPHA))
    runs["c_retrain_recalibrate"].append(report(m2.score(probes.text.values), m2.score(btest.text.values), thr_c, btest))
    # (d) + synthetic Indian scam / look-alike templates (synth_india.py), then (c)
    syn = generate(seed=seed)
    X = np.concatenate([dtr.text.values, b_tr.text.values, syn.text.values])
    y = np.concatenate([dtr.is_smish.values, np.zeros(len(b_tr), int), syn.is_smish.values])
    m3 = MessageModel("logreg").fit(X, y)
    s_dcal = m3.score(dcal.text.values)
    thr_d = max(conformal_threshold(s_dcal[dcal.is_smish.values == 0], ALPHA),
                conformal_threshold(m3.score(b_cal.text.values), ALPHA))
    rep = report(m3.score(probes.text.values), m3.score(btest.text.values), thr_d, btest)
    rep["mishra_recall_kept"] = float((s_dcal[dcal.is_smish.values == 1] > thr_d).mean())
    rep["mishra_false_alert"] = float((s_dcal[dcal.is_smish.values == 0] > thr_d).mean())
    runs["d_plus_synthetic_india"].append(rep)
    s_m_cal = m.score(dcal.text.values)
    runs["a_base"][-1]["mishra_recall_kept"] = float((s_m_cal[dcal.is_smish.values == 1] > thr_a).mean())
    runs["a_base"][-1]["mishra_false_alert"] = float((s_m_cal[dcal.is_smish.values == 0] > thr_a).mean())
    print(seed, {k: round(v[-1]["scam_recall_all"], 3) for k, v in runs.items()},
          {k: round(v[-1]["indian_benign_false_alert"], 3) for k, v in runs.items()})


def mean_dict(lst):
    out = {}
    for k in lst[0]:
        if isinstance(lst[0][k], dict):
            keys = set().union(*[d[k].keys() for d in lst])
            out[k] = {kk: float(np.nanmean([d[k].get(kk, np.nan) for d in lst])) for kk in keys}
            continue
        if False:
            out[k] = {kk: float(np.mean([d[k].get(kk, np.nan) for d in lst])) for kk in lst[0][k]}
        else:
            out[k] = float(np.mean([d[k] for d in lst]))
            out[k + "_sd"] = float(np.std([d[k] for d in lst]))
    return out


R["results"] = {k: mean_dict(v) for k, v in runs.items()}
R["note"] = ("(a) Mishra only; (b) threshold re-set on unlabelled Indian non-scam traffic; "
             "(c) that traffic also used as training negatives; (d) plus synthetic Indian scam/look-alike templates. No Indian scam message is ever used "
             "for training or calibration. Probes = hand-written Indian scams + legitimate look-alikes.")
json.dump(R, open("results/external.json", "w"), indent=2, default=float)
print(json.dumps(R["results"], indent=1))
