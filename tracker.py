"""Journey tracker: turns consented on-device events into kill-chain stage
scores and the feature vector the risk engine sees just before the UPI PIN
pad opens.

Every feature below maps to a signal a bank / PSP app can legally read on
Android 11+ (see SIGNAL_SOURCES). The SDK never reads call logs, contacts or
the caller's number: it only knows whether a call is active while the app is
open (AudioManager mode, no permission) and whether that call is VoIP.

Stage score with time decay (one value per stage k):
    s_k(t) = max over evidence e of  w_e * exp(-(t - t_e) / h_k)
Evidence can only push a stage up (monotonic); it fades on its own.
On the phone this is a small Kotlin state machine; here it is the same logic
as a plain Python function.
"""
import math

DECAY_H = {"s1": 7 * 24, "s2": 3 * 24, "s3": 3 * 24}   # decay scale, hours

SIGNAL_SOURCES = {
    "on_call": "AudioManager.getMode() == IN_CALL / IN_COMMUNICATION (no permission)",
    "voip_call": "AudioManager mode IN_COMMUNICATION = VoIP / WhatsApp-style call",
    "remote_app_2h": "<queries> for named remote-access apps + firstInstallTime; Android 15 "
                     "screen-recording callback; enabled non-system accessibility services",
    "msg_risk_session": "message the user chose to share into the app",
    "payment features": "payment session data the bank / PSP already holds",
    "unknown_credits_7d": "incoming credits from first-time senders (bank's own ledger)",
}

SESSION_FEATURES = ["log_amount", "amount_ratio", "new_payee", "secs_to_pin", "night",
                    "payee_reported", "on_call", "voip_call", "remote_app_2h", "msg_risk_session"]
JOURNEY_FEATURES = ["s1", "s2", "s3", "n_newpayee_7d", "growth", "cum_newpayee_ratio_7d",
                    "unknown_credits_7d", "n_pay_on_call_7d"]
PAYMENT_ONLY = ["log_amount", "amount_ratio", "new_payee", "secs_to_pin", "night", "payee_reported"]


def _decay(w, dt, k):
    return w * math.exp(-dt / DECAY_H[k])


def extract(j, msg_score):
    """msg_score: dict id(event) -> P(scam) from the guarded message model."""
    ev = []                       # (t, stage, weight)
    pays, credits, calls, apps = [], [], [], []
    rows = []
    for e in j.events:
        t = e.t
        if e.kind == "msg" and e.data["shared"]:
            ev.append((t, "s1", msg_score[id(e)]))
        elif e.kind == "call":
            calls.append((t, e.data))          # ground truth only; the SDK sees it only at app sessions
        elif e.kind == "app" and e.data.get("remote"):
            apps.append(t); ev.append((t, "s3", 0.9))
        elif e.kind == "credit" and e.data.get("unknown"):
            credits.append(t); ev.append((t, "s3", 0.5))
        elif e.kind == "pay":
            d = e.data
            active = [c for ts, c in calls if ts <= t <= ts + c["dur"] / 60]
            on_call, voip = bool(active), any(c.get("voip") for c in active)
            if on_call:                        # pressure is observed only when the app is open
                ev.append((t, "s2", 1.0 if voip else 0.7))
            st = {k: 0.0 for k in DECAY_H}
            for te, k, w in ev:
                if te <= t:
                    st[k] = max(st[k], _decay(w, t - te, k))
            np7 = [(tp, a) for tp, a, newp, _ in pays if newp and t - tp <= 7 * 24]
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
                "voip_call": int(voip),
                "remote_app_2h": int(any(0 <= t - ta <= 2 for ta in apps)),
                "msg_risk_session": max(sess_msgs, default=0.0),
                **st,
                "n_newpayee_7d": len(np7),
                "growth": (d["amount"] / np7[-1][1]) if (np7 and d["new_payee"]) else 0.0,
                "cum_newpayee_ratio_7d": sum(a for _, a in np7) / j.median_amt,
                "unknown_credits_7d": sum(1 for tc in credits if 0 <= t - tc <= 7 * 24),
                "n_pay_on_call_7d": sum(1 for tp, _, _, oc in pays if oc and t - tp <= 7 * 24),
                # metadata (not model inputs)
                "_y": int(d["scam"]), "_amount": d["amount"], "_user": j.user, "_age": j.age,
                "_lang": j.lang, "_label": j.label, "_hard": j.hard_neg, "_hardpay": int(d.get("hard", False)),
            }
            rows.append(row)
            pays.append((t, d["amount"], d["new_payee"], on_call))
    return rows


def families(r, msg_thr=0.5):
    """Independent evidence families for the two-family rule (Tier-3 holds)."""
    f = set()
    if r["msg_risk_session"] > msg_thr or r["s1"] > msg_thr:
        f.add("message")
    if r["on_call"]:
        f.add("call")
    if r["remote_app_2h"]:
        f.add("device")
    if r["new_payee"] and (r["amount_ratio"] > 3 or r["secs_to_pin"] < 12):
        f.add("payment")
    if r["s3"] > 0.4 or (r["n_newpayee_7d"] >= 2 and r["growth"] > 1.5) or r["n_pay_on_call_7d"] >= 2:
        f.add("journey")
    if r["payee_reported"]:
        f.add("intel")
    return f
