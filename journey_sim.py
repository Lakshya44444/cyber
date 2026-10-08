"""Synthetic scam / genuine payment journeys for the journey-level benchmark.

No real bank data exists publicly, so the *journey structure* (calls, payees,
amounts, timing) is simulated from public RBI / I4C scam typologies. The *text*
of every message a user shares is a REAL message from the held-out part of the
public SMS dataset (smishing texts for scams; ham / spam for genuine users), so
the message model is scored on text it has never seen.

Hard negatives are built in on purpose (hospital payment during a family call,
rent deposit to a new landlord, paying a relative while a son explains on the
phone, a real broker, a work remote-desktop app) because those are what make
blanket rules noisy.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

AGE_BANDS = ["18-35", "36-59", "60+"]
LANGS = ["en", "hi"]


@dataclass
class Event:
    t: float            # hours since start of window
    kind: str           # msg | call | app | credit | contact | pay
    data: dict = field(default_factory=dict)


@dataclass
class Journey:
    user: int
    age: str
    lang: str
    label: str          # genuine | task_scam | digital_arrest | kyc_scam
    hard_neg: str       # name of hard-negative scenario, or ""
    median_amt: float
    events: list[Event] = field(default_factory=list)


class Simulator:
    """signal: multiplier on every scam-side signal probability (call, fast PIN,
    remote app, reported payee, new payee, payouts). share: multiplier on the
    chance a scam message is shared. hard: multiplier on the hard-negative rate.
    Defaults (1, 1, 1) are the main benchmark; the sensitivity sweep varies them."""

    def __init__(self, scam_texts, ham_texts, spam_texts, seed=0, signal=1.0, share=1.0, hard=1.0):
        self.r = random.Random(seed)
        self.scam_texts, self.ham_texts, self.spam_texts = scam_texts, ham_texts, spam_texts
        self.sig, self.share, self.hard = signal, share, hard

    def p(self, x):            # scaled scam-signal probability
        return self.r.random() < min(1.0, x * self.sig)

    # ------------------------------------------------------------ helpers
    def _msg(self, t, pool, shared_p, adversarial_p=0.0):
        r = self.r
        return Event(t, "msg", {"text": r.choice(pool), "shared": r.random() < shared_p,
                                "adv": r.random() < adversarial_p})

    def _pay(self, t, amt, new_payee, reported=False, fast=False, scam=False):
        r = self.r
        secs = r.uniform(8, 25) if fast else r.uniform(15, 90)
        return Event(t, "pay", {"amount": round(amt), "new_payee": new_payee,
                                "reported": reported, "secs_to_pin": secs, "scam": scam})

    def _background(self, j: Journey, days=30):
        """Ordinary life: routine payments, chats, marketing SMS, family calls."""
        r = self.r
        n = r.randint(6, 14)
        for _ in range(n):
            t = r.uniform(0, days * 24)
            known = r.random() < 0.85
            amt = j.median_amt * math.exp(r.gauss(0, 0.5)) if known else r.uniform(100, 3000)
            j.events.append(self._pay(t, amt, new_payee=not known))
        for _ in range(r.randint(0, 3)):  # genuine messages the user chose to share
            j.events.append(self._msg(r.uniform(0, days * 24), self.ham_texts if r.random() < .6 else self.spam_texts, 1.0))
        for _ in range(r.randint(1, 6)):  # calls: family, delivery, customer care, bank
            t = r.uniform(0, days * 24)
            unknown = r.random() < .3
            dur = r.uniform(5, 45) if (unknown and r.random() < .25) else r.uniform(1, 25)
            j.events.append(Event(t, "call", {"dur": dur, "unknown": unknown, "voip": r.random() < .4}))
            if unknown and r.random() < .3:
                j.events.append(Event(t, "contact", {"unknown": True}))
        if r.random() < .35:              # cashback, refunds, friends paying back
            for _ in range(r.randint(1, 3)):
                j.events.append(Event(r.uniform(0, days * 24), "credit", {"amt": r.uniform(20, 600), "unknown": True}))
        if r.random() < .15:              # shopping spree / wedding / bill split: several new payees, growing
            t = r.uniform(0, (days - 7) * 24)
            amt = j.median_amt * r.uniform(.5, 2)
            for _ in range(r.randint(2, 4)):
                j.events.append(self._pay(t, amt, True, fast=r.random() < .3))
                amt *= r.uniform(1.0, 2.5)
                t += r.uniform(3, 40)
        if r.random() < .01:
            j.events.append(Event(r.uniform(0, days * 24), "app", {"remote": True}))

    # ------------------------------------------------------------ genuine hard negatives
    def _hard_negative(self, j: Journey, kind: str):
        r, end = self.r, 30 * 24
        t = r.uniform(5 * 24, end - 1)
        m = j.median_amt
        if kind == "hospital_on_family_call":
            j.events.append(Event(t - .3, "call", {"dur": r.uniform(20, 60), "unknown": False, "voip": r.random() < .5}))
            j.events.append(self._pay(t, m * r.uniform(8, 30), True, fast=r.random() < .5))
        elif kind == "rent_deposit_new_landlord":
            if r.random() < .5:
                j.events.append(Event(t - .1, "call", {"dur": r.uniform(5, 20), "unknown": True, "voip": r.random() < .3}))
            j.events.append(self._pay(t, m * r.uniform(6, 20), True))
        elif kind == "relative_with_son_on_call":
            j.events.append(Event(t - .5, "call", {"dur": r.uniform(30, 70), "unknown": False, "voip": r.random() < .6}))
            j.events.append(self._pay(t, m * r.uniform(3, 12), True))
        elif kind == "registered_broker":
            j.events.append(self._msg(t - 48, self.spam_texts, .5))
            j.events.append(self._pay(t, m * r.uniform(5, 25), True))
        elif kind == "work_remote_desktop":
            j.events.append(Event(t - 1, "app", {"remote": True}))
            j.events.append(self._pay(t, m * r.uniform(1, 4), r.random() < .5))
        j.events[-1].data["hard"] = True

    # ------------------------------------------------------------ scam journeys
    def _task_scam(self, j: Journey):
        r, P, sh = self.r, self.p, self.share
        t0 = r.uniform(0, 12 * 24)
        if r.random() < .5:                         # first contact often on WhatsApp: not visible
            j.events.append(Event(t0, "contact", {"unknown": True}))
        j.events.append(self._msg(t0 + .1, self.scam_texts, .3 * sh, .3))
        t = t0 + r.uniform(4, 30)
        n_pay = r.choice([0, 0, 1, 2, 3]) if self.sig >= 1 else (r.choice([0, 0, 1, 2, 3]) if P(1.0) else 0)
        for _ in range(n_pay):                      # small payouts build trust (S3), not always
            j.events.append(Event(t, "credit", {"amt": r.uniform(80, 600), "unknown": True}))
            t += r.uniform(2, 20)
        amt = r.uniform(800, 3000)
        for k in range(r.randint(3, 6)):            # loop S3 -> S4 with growing amounts
            if P(.35):
                j.events.append(Event(t - .2, "call", {"dur": r.uniform(10, 60), "unknown": True, "voip": r.random() < .8}))
            if k and r.random() < .3:
                j.events.append(self._msg(t - 1, self.scam_texts, .3 * sh, .3))
            j.events.append(self._pay(t, amt, new_payee=P(.7), reported=P(.1),
                                      fast=P(.6), scam=True))
            amt *= r.uniform(1.3, 3.0)
            t += r.uniform(6, 48)
            if t > 30 * 24:
                break

    def _digital_arrest(self, j: Journey):
        r, P, sh = self.r, self.p, self.share
        t0 = r.uniform(2 * 24, 28 * 24)
        j.events.append(Event(t0, "contact", {"unknown": True}))
        if r.random() < .3:
            j.events.append(self._msg(t0 + .05, self.scam_texts, .35 * sh, .2))
        dur = r.uniform(60, 240)
        j.events.append(Event(t0 + .05, "call", {"dur": dur, "unknown": True, "video": P(1.0), "voip": True}))
        if P(.4):                                   # "share your screen so the officer can verify"
            j.events.append(Event(t0 + .3, "app", {"remote": True}))
        for k in range(r.randint(1, 2)):
            j.events.append(self._pay(t0 + .5 + k * .4, j.median_amt * r.uniform(10, 60), P(1.0),
                                      reported=P(.1), fast=P(.4), scam=True))

    def _kyc_scam(self, j: Journey):
        r, P, sh = self.r, self.p, self.share
        t0 = r.uniform(0, 28 * 24)
        j.events.append(self._msg(t0, self.scam_texts, .35 * sh, .3))
        if r.random() < .6:
            j.events.append(Event(t0 + .2, "contact", {"unknown": True}))
        if P(.6):
            j.events.append(Event(t0 + .3, "call", {"dur": r.uniform(15, 50), "unknown": True, "voip": r.random() < .3}))
        if P(.5):
            j.events.append(Event(t0 + .4, "app", {"remote": True}))
        j.events.append(self._pay(t0 + .6, j.median_amt * r.uniform(2, 25), P(1.0),
                                  reported=P(.1), fast=P(.5), scam=True))

    # ------------------------------------------------------------ users
    def user(self, uid: int, scam_rate: float) -> Journey:
        r = self.r
        age = r.choices(AGE_BANDS, [.4, .38, .22])[0]
        lang = r.choices(LANGS, [.55, .45])[0]
        med = math.exp(r.gauss(math.log(2500), .6)) * (1.2 if age == "36-59" else 1.0)
        label = "genuine"
        if r.random() < scam_rate:
            label = r.choices(["task_scam", "digital_arrest", "kyc_scam"], [.77, .08, .15])[0]
        hard = ""
        # elderly and Hindi-first users get more "help on the phone" payments
        hn_p = .35 + (.25 if age == "60+" else 0) + (.1 if lang == "hi" else 0)
        if r.random() < min(.95, hn_p * self.hard):
            w = [3 if age == "60+" else 1, 1, 3 if age == "60+" else 1, 1, .5]
            hard = r.choices(["hospital_on_family_call", "rent_deposit_new_landlord",
                              "relative_with_son_on_call", "registered_broker",
                              "work_remote_desktop"], w)[0]
        j = Journey(uid, age, lang, label, hard, med)
        self._background(j)
        if hard:
            self._hard_negative(j, hard)
        if label == "task_scam":
            self._task_scam(j)
        elif label == "digital_arrest":
            self._digital_arrest(j)
        elif label == "kyc_scam":
            self._kyc_scam(j)
        j.events.sort(key=lambda e: e.t)
        return j
