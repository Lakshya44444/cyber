"""Journey-level benchmark. Writes results/journey.json and results/examples.json."""
import json
import random
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

from attacks import ATTACKS
from conformal import conformal_threshold
from data import load_smishing
from guardrails import TEMPLATE, check_explanation, explanation_or_fallback
from journey_sim import Simulator
from message_model import MessageModel
from tracker import (JOURNEY_FEATURES, PAYMENT_ONLY, SESSION_FEATURES, extract, families)

import os
SEED, N_USERS, SCAM_RATE = 5, int(os.environ.get("N_USERS", 16000)), 0.15
SIM_KNOBS = {k: float(os.environ.get(k.upper(), 1.0)) for k in ("signal", "share", "hard")}
OUT = os.environ.get("OUT", "results/journey.json")
A1, A2, A3 = 0.10, 0.02, 0.005          # tier budgets (false-alert rate on genuine payments)
rng = random.Random(SEED)
R = {"setup": {"users": N_USERS, "scam_user_share_in_sim": SCAM_RATE,
               "alpha_T1": A1, "alpha_T2": A2, "alpha_T3": A3}}

# ---------------------------------------------- 1. message model on one half of the REAL texts
df = load_smishing()
tr_i, sim_i = next(GroupShuffleSplit(1, test_size=0.4, random_state=SEED)
                   .split(df, df["is_smish"], df["group"]))
mtrain, msim = df.iloc[tr_i], df.iloc[sim_i]
mm = MessageModel("logreg").fit(mtrain["text"].values, mtrain["is_smish"].values)
pools = {k: msim.loc[msim["label"] == k, "text"].tolist() for k in ["smishing", "ham", "spam"]}
R["setup"]["held_out_texts_used_in_journeys"] = {k: len(v) for k, v in pools.items()}

# ---------------------------------------------- 2. simulate journeys
sim = Simulator(pools["smishing"], pools["ham"], pools["spam"], seed=SEED, **SIM_KNOBS)
R["setup"]["sim_knobs"] = SIM_KNOBS
journeys = [sim.user(u, SCAM_RATE) for u in range(N_USERS)]
msgs = [e for j in journeys for e in j.events if e.kind == "msg" and e.data["shared"]]
attack_names = list(ATTACKS)
texts = [ATTACKS[rng.choice(attack_names)](e.data["text"], rng) if e.data["adv"] else e.data["text"]
         for e in msgs]
scores = mm.score(texts)
msg_score = {id(e): float(s) for e, s in zip(msgs, scores)}
rows = [r for j in journeys for r in extract(j, msg_score)]
D = pd.DataFrame(rows)
D = D[D["_amount"] >= 1000].reset_index(drop=True)      # PIN-screen events worth scoring
D["_group"] = D["_age"] + "|" + D["_lang"]
R["setup"]["payment_events"] = int(len(D))
R["setup"]["scam_payment_events"] = int(D["_y"].sum())
R["setup"]["shared_messages"] = len(msgs)
R["setup"]["adversarial_messages"] = int(sum(e.data["adv"] for e in msgs))

users = D["_user"].unique()
rng.shuffle(users := list(users))
n = len(users)
split = {u: ("train" if i < .5 * n else "cal" if i < .75 * n else "test") for i, u in enumerate(users)}
D["_split"] = D["_user"].map(split)
TR, CA, TE = (D[D["_split"] == s] for s in ["train", "cal", "test"])


def gbm(cols):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                       l2_regularization=1.0, random_state=SEED)
    return m.fit(TR[cols], TR["_y"])


def evaluate(alert, df):
    y = df["_y"].values.astype(bool)
    g = ~y
    out = {
        "scam_payment_recall": float(alert[y].mean()),
        "scam_rupees_covered": float(df.loc[y & alert, "_amount"].sum() / df.loc[y, "_amount"].sum()),
        "false_alert_rate_genuine": float(alert[g].mean()),
        "false_alert_rate_hard_negatives": float(alert[g & (df["_hardpay"] == 1).values].mean()),
    }
    jd = df.assign(a=alert).query("_y == 1").groupby("_user")["a"].any()
    out["scam_journeys_flagged"] = float(jd.mean())
    by = {}
    for grp, sub in df.assign(a=alert)[g].groupby("_group"):
        by[grp] = float(sub["a"].mean())
    out["false_alert_by_group"] = by
    out["subgroup_gap_pts"] = float((max(by.values()) - min(by.values())) * 100)
    return out


R["methods"] = {}
# B1. RBI-style blanket rule: every new-payee payment above Rs 10,000 gets friction
R["methods"]["RBI rule (new payee > ₹10k)"] = evaluate(
    ((TE["new_payee"] == 1) & (TE["_amount"] > 10000)).values, TE)

# B2. single-message classifier (only helps when a message was shared)
thr = conformal_threshold(CA.loc[CA["_y"] == 0, "msg_risk_session"], A2)
R["methods"]["Single-SMS classifier"] = evaluate((TE["msg_risk_session"] > thr).values, TE)

# B3 / B4 / full model
cols_full = SESSION_FEATURES + JOURNEY_FEATURES
models = {"Payment-only model": PAYMENT_ONLY, "ScamTrail without journey state": SESSION_FEATURES,
          "ScamTrail (full)": cols_full}
fitted = {}
for name, cols in models.items():
    t0 = time.perf_counter()
    m = gbm(cols)
    fitted[name] = (m, cols)
    s_ca = m.predict_proba(CA[cols])[:, 1]
    s_te = m.predict_proba(TE[cols])[:, 1]
    t = conformal_threshold(s_ca[CA["_y"].values == 0], A2)   # same 2% budget for every model
    R["methods"][name] = evaluate(s_te > t, TE)


# ---------------------------------------------- 3. full ScamTrail decision: Mondrian tiers + two-family rule
m, cols = fitted["ScamTrail (full)"]
s_ca, s_te = m.predict_proba(CA[cols])[:, 1], m.predict_proba(TE[cols])[:, 1]


def tiers(scores, df, mondrian=True, two_family=True, cal_scores=s_ca, cal=CA, two_family_from=2):
    groups = df["_group"].values if mondrian else np.array(["all"] * len(df))
    cgroups = cal["_group"].values if mondrian else np.array(["all"] * len(cal))
    th = {}
    for g in np.unique(groups):
        neg = cal_scores[(cgroups == g) & (cal["_y"].values == 0)]
        th[g] = [conformal_threshold(neg, a) for a in (A1, A2, A3)]
    tier = np.zeros(len(df), int)
    fams = [families(r) for r in df[cols].to_dict("records")]
    for i, (s, g) in enumerate(zip(scores, groups)):
        t1, t2, t3 = th[g]
        k = 3 if s > t3 else 2 if s > t2 else 1 if s > t1 else 0
        if two_family and k >= two_family_from and len(fams[i]) < 2:
            k = two_family_from - 1      # one family alone can never trigger friction at/above this tier
        tier[i] = k
    return tier


T = tiers(s_te, TE)
R["decision"] = evaluate(T >= 2, TE)
R["ablations"] = {
    "full decision (Mondrian + two-family)": R["decision"],
    "no_mondrian (one global threshold)": evaluate(tiers(s_te, TE, mondrian=False) >= 2, TE),
    "no_two_family_rule": evaluate(tiers(s_te, TE, two_family=False) >= 2, TE),
}
R["ablations"]["mondrian_only (no two-family)"] = R["ablations"]["no_two_family_rule"]
R["ablations"]["two-family rule only for T3 holds"] = evaluate(tiers(s_te, TE, two_family_from=3) >= 2, TE)
# hold-level view: Tier 3 (payment held) is where friction is costly
T3only = tiers(s_te, TE, two_family_from=3)
R["holds_T3"] = {"two_family_all_tiers": evaluate(T >= 3, TE), "two_family_T3_only": evaluate(T3only >= 3, TE)}
R["ablations"]["global_only (no two-family)"] = evaluate(tiers(s_te, TE, mondrian=False, two_family=False) >= 2, TE)
# works without messages: user never shares anything at test time
TEn = TE.copy()
TEn["msg_risk_session"] = 0.0
TEn["s1"] = 0.0
R["ablations"]["no messages shared at all"] = evaluate(tiers(m.predict_proba(TEn[cols])[:, 1], TEn) >= 2, TEn)

R["tier_distribution"] = {
    "genuine": {f"T{k}": float((T[TE["_y"].values == 0] == k).mean()) for k in range(4)},
    "scam": {f"T{k}": float((T[TE["_y"].values == 1] == k).mean()) for k in range(4)},
}
R["recall_by_scam_type"] = {lab: float((T[(TE["_label"] == lab).values & (TE["_y"] == 1).values] >= 2).mean())
                            for lab in ["task_scam", "digital_arrest", "kyc_scam"]}

# latency: one PIN-screen decision = feature row -> score -> tier
row = TE[cols].iloc[[0]]
t0 = time.perf_counter()
for _ in range(300):
    m.predict_proba(row)
R["latency_ms_per_decision"] = float((time.perf_counter() - t0) / 300 * 1e3)

# ---------------------------------------------- 4. reasons (occlusion) -> explanation -> guard
REASON_OF = {"on_call": "on_call", "call_minutes": "on_call", "call_unknown": "on_call", "s2": "on_call",
             "new_payee": "new_payee", "amount_ratio": "amount_high", "log_amount": "amount_high",
             "remote_app_2h": "remote_app", "msg_risk_session": "msg_scam_like", "s1": "msg_scam_like",
             "secs_to_pin": "fast_pin", "n_newpayee_7d": "repeat_loop", "growth": "repeat_loop",
             "cum_newpayee_ratio_7d": "repeat_loop", "s3": "repeat_loop", "unknown_credits_7d": "repeat_loop",
             "payee_reported": "payee_reported"}
baseline = TR.loc[TR["_y"] == 0, cols].median()


def reasons(x):
    base = m.predict_proba(x[cols])[:, 1][0]
    drops = {}
    for c in cols:
        if c not in REASON_OF:
            continue
        z = x[cols].copy()
        z[c] = baseline[c]
        drops[REASON_OF[c]] = max(drops.get(REASON_OF[c], 0), base - m.predict_proba(z)[:, 1][0])
    return [k for k, v in sorted(drops.items(), key=lambda kv: -kv[1]) if v > 0.02][:3]


examples = []
for idx in np.where((T >= 2) & (TE["_y"].values == 1))[0][:4].tolist() + \
           np.where((T >= 2) & (TE["_y"].values == 0))[0][:2].tolist():
    x = TE.iloc[[idx]]
    rs = reasons(x)
    cand = " ".join(TEMPLATE[r] for r in rs[:2]) + " Please pause and check before you pay."
    shown, viol = explanation_or_fallback(cand, rs)
    examples.append({"truth": "scam" if x["_y"].iloc[0] else "genuine", "type": x["_label"].iloc[0],
                     "hard_negative": x["_hard"].iloc[0], "tier": int(T[idx]),
                     "amount": int(x["_amount"].iloc[0]), "reasons": rs, "families": sorted(families(x[cols].iloc[0].to_dict())),
                     "shown": shown, "guard_violations": viol})
R["examples"] = examples

# ---------------------------------------------- 5. explanation guard red-team (simulated LLM failure modes)
red = []
r2 = random.Random(1)
codes = list(TEMPLATE)
for _ in range(400):
    rs = r2.sample(codes, 2)
    good = " ".join(TEMPLATE[r] for r in rs) + " Please pause and check."
    other = r2.choice([c for c in codes if c not in rs])
    bad = {
        "ungrounded_reason": good + " " + TEMPLATE[other],
        "says_safe": "This payee looks safe, but " + TEMPLATE[rs[0]].lower(),
        "accuses_user_contact": TEMPLATE[rs[0]] + " You are being scammed by a fraudster.",
        "leaks_score": TEMPLATE[rs[0]] + " Risk score 0.93.",
        "threat": TEMPLATE[rs[0]] + " Your account will be blocked if you continue.",
        "too_long": good + " " + " ".join([TEMPLATE[rs[0]]] * 6),
    }
    red.append(("good", not check_explanation(good, rs)))
    for k, txt in bad.items():
        red.append((k, bool(check_explanation(txt, rs))))
rt = pd.DataFrame(red, columns=["case", "ok"])
R["explanation_guard"] = {c: float(g["ok"].mean()) for c, g in rt.groupby("case")}
R["explanation_guard_note"] = "'good' = share of valid explanations allowed through; others = share of bad ones blocked"

json.dump(R, open(OUT, "w"), indent=2, default=float)
summ = {k: {kk: round(vv, 3) for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in R["methods"].items()}
print(json.dumps(R["setup"], indent=1))
print(pd.DataFrame(summ).T.to_string())
print("decision:", {k: (round(v, 4) if not isinstance(v, dict) else v) for k, v in R["decision"].items()})
print("ablations:", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in R["ablations"].items()}, indent=1))
print("tiers:", R["tier_distribution"])
print("by type:", R["recall_by_scam_type"])
print("latency ms:", R["latency_ms_per_decision"])
print("guard:", R["explanation_guard"])
for k in ["no_mondrian (one global threshold)", "full decision (Mondrian + two-family)"]:
    print("groups", k, {g: round(v, 4) for g, v in R["ablations"][k]["false_alert_by_group"].items()})
for e in examples:
    print(e)
