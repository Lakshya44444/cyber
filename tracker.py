"""Journey tracker: turns a stream of consented on-device events into
kill-chain stage scores and the feature vector the risk engine sees at the
UPI PIN screen.

Stage score with time decay (one value per stage k):
    s_k(t) = max over evidence e of  w_e * exp(-lambda_k * (t - t_e))
Evidence can only push a stage up (monotonic); it fades on its own.
In the product this state lives in a small LangGraph graph on the phone;
here it is the same logic as a plain Python function.
"""
import math

DECAY_H = {"s0": 72, "s1": 7 * 24, "s2": 6, "s3": 3 * 24}  # half-life-ish, hours

SESSION_FEATURES = ["log_amount", "amount_ratio", "new_payee", "secs_to_pin", "night",
                    "payee_reported", "on_call", "call_minutes", "call_unknown",
                    "remote_app_2h", "msg_risk_session"]
JOURNEY_FEATURES = ["s0", "s1", "s2", "s3", "n_newpayee_7d", "growth", "cum_newpayee_ratio_7d",
                    "unknown_credits_7d", "hours_since_contact"]
PAYMENT_ONLY = ["log_amount", "amount_ratio", "new_payee", "secs_to_pin", "night", "payee_reported"]
MESSAGE_FEATS = ["msg_risk_session", "s1"]


def _decay(w, dt, k):
    return w * math.exp(-dt / DECAY_H[k])


def extract(j, msg_score):
    """msg_score: dict id(event) -> P(smishing) from the guarded message model."""
    ev = []          # (t, stage, weight)
    pays, credits, calls, apps = [], [], [], []
    last_contact = None
    rows = []
    for e in j.events:
        t = e.t
        if e.kind == "contact":
            ev.append((t, "s0", 0.6)); last_contact = t
        elif e.kind == "msg" and e.data["shared"]:
            ev.append((t, "s1", msg_score[id(e)]))
            if msg_score[id(e)] > .5:
                last_contact = t if last_contact is None else last_contact
        elif e.kind == "call":
            calls.append((t, e.data))
            if e.data.get("unknown"):
                ev.append((t, "s0", 0.4)); last_contact = last_contact or t
                ev.append((t, "s2", 1.0 if e.data.get("video") else min(1.0, e.data["dur"] / 60)))
        elif e.kind == "app" and e.data.get("remote"):
            apps.append(t); ev.append((t, "s3", 0.9))
        elif e.kind == "credit" and e.data.get("unknown"):
            credits.append(t); ev.append((t, "s3", 0.5))
        elif e.kind == "pay":
            d = e.data
            st = {k: 0.0 for k in DECAY_H}
            for te, k, w in ev:
                st[k] = max(st[k], _decay(w, t - te, k))
            active = [(ts, c) for ts, c in calls if ts <= t <= ts + c["dur"] / 60]
            on_call = bool(active)
            np7 = [(tp, a) for tp, a, newp in pays if newp and t - tp <= 7 * 24]
            sess_msgs = [msg_score[id(m)] for m in j.events
                         if m.kind == "msg" and m.data["shared"] and 0 <= t - m.t <= 1]
            row = {
                "log_amount": math.log10(d["amount"] + 1),
                "amount_ratio": d["amount"] / j.median_amt,
                "new_payee": int(d["new_payee"]),
                "secs_to_pin": d["secs_to_pin"],
                "night": int((t % 24) < 6),
                "payee_reported": int(d["reported"]),
                "on_call": int(on_call),
                "call_minutes": max(((t - ts) * 60 for ts, _ in active), default=0.0),
                "call_unknown": int(any(c.get("unknown") for _, c in active)),
                "remote_app_2h": int(any(0 <= t - ta <= 2 for ta in apps)),
                "msg_risk_session": max(sess_msgs, default=0.0),
                **st,
                "n_newpayee_7d": len(np7),
                "growth": (d["amount"] / np7[-1][1]) if (np7 and d["new_payee"]) else 0.0,
                "cum_newpayee_ratio_7d": sum(a for _, a in np7) / j.median_amt,
                "unknown_credits_7d": sum(1 for tc in credits if t - tc <= 7 * 24),
                "hours_since_contact": (t - last_contact) if last_contact is not None else 999.0,
                # metadata (not model inputs)
                "_y": int(d["scam"]), "_amount": d["amount"], "_user": j.user, "_age": j.age,
                "_lang": j.lang, "_label": j.label, "_hard": j.hard_neg, "_hardpay": int(d.get("hard", False)),
            }
            rows.append(row)
            pays.append((t, d["amount"], d["new_payee"]))
    return rows


def families(r, msg_thr=0.5):
    """Independent evidence families for the two-family rule."""
    f = set()
    if r["msg_risk_session"] > msg_thr or r["s1"] > msg_thr:
        f.add("message")
    if r["on_call"] and (r["call_unknown"] or r["call_minutes"] > 20):
        f.add("call")
    if r["remote_app_2h"]:
        f.add("device")
    if r["new_payee"] and (r["amount_ratio"] > 3 or r["secs_to_pin"] < 12):
        f.add("payment")
    if r["s3"] > 0.4 or (r["n_newpayee_7d"] >= 2 and r["growth"] > 1.5):
        f.add("journey")
    if r["payee_reported"]:
        f.add("intel")
    return f
