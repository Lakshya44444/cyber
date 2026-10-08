"""Make the result figures used in the deck (PNG, 300 dpi, white card background)."""
import json

import matplotlib.pyplot as plt

M = json.load(open("results/message.json"))
J = json.load(open("results/journey.json"))
NAVY, BLUE, GRAY, ORANGE, INK2 = "#102A43", "#1F5FAD", "#7B8794", "#E4572E", "#52606D"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": "#CBD2D9",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(f"results/{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. same false-alert budget: who catches the scam payments? (synthetic journeys)
order = ["RBI rule (new payee > ₹10k)", "Single-SMS classifier", "Payment-only model",
         "ScamTrail without journey state", "ScamTrail (full)"]
labels = ["Blanket rule\n(new payee > ₹10k)*", "Single-SMS\nclassifier", "Payment-only\nmodel",
          "ScamTrail,\nno journey state", "ScamTrail\n(full)"]
rec = [J["methods"][k]["scam_payment_recall"] * 100 for k in order]
hard = [J["methods"][k]["false_alert_rate_hard_negatives"] * 100 for k in order]
for name, vals, title, fmt in [
    ("fig_recall", rec, "Scam payments caught at the same 2% false-alert budget", "{:.0f}%"),
    ("fig_hardneg", hard, "False alerts on hard genuine cases\n(hospital on a family call, new landlord, …)", "{:.0f}%")]:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    cols = [BLUE if "full" in k else GRAY for k in order]
    bars = ax.barh(labels, vals, color=cols, height=0.62)
    for b, v in zip(bars, vals):
        ax.text(v + 1.2, b.get_y() + b.get_height() / 2, fmt.format(v), va="center",
                fontsize=13, color=NAVY, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(title, loc="left", fontsize=14, color=NAVY, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    fig.text(0.0, -0.04, "* blanket rule flags 5% of genuine payments, not 2%.  Synthetic journeys; message text is real held-out SMS.",
             fontsize=9.5, color=INK2)
    save(fig, name)

# 2. conformal budget holds on REAL messages
fig, ax = plt.subplots(figsize=(5.2, 4.2))
alphas = sorted(M["conformal"], key=float)
tgt = [float(a) * 100 for a in alphas]
obs = [M["conformal"][a]["observed_fpr_mean"] * 100 for a in alphas]
p90 = [M["conformal"][a]["observed_fpr_p90"] * 100 for a in alphas]
ax.plot([0, 6], [0, 6], color="#CBD2D9", lw=2, ls="--", zorder=1)
ax.vlines(tgt, obs, p90, color=BLUE, lw=2, alpha=.5, zorder=2)
ax.scatter(tgt, obs, s=90, color=BLUE, zorder=3, edgecolor="white", linewidth=2)
for x, y_, a in zip(tgt, obs, alphas):
    r = M["conformal"][a]["recall_mean"] * 100
    ax.annotate(f"{y_:.1f}% observed\n{r:.0f}% scams caught", (x, y_), xytext=(10, -26),
                textcoords="offset points", fontsize=10.5, color=NAVY)
ax.set_xlim(0, 6.2); ax.set_ylim(0, 7)
ax.set_xlabel("False-alert budget we set (%)")
ax.set_ylabel("False alerts measured (%)")
ax.set_title("The alert budget holds on real SMS", loc="left", fontsize=14, color=NAVY, fontweight="bold")
fig.text(0.0, -0.06, "20 random splits; dot = mean, line = 90th percentile. Dashed = perfect match.",
         fontsize=9.5, color=INK2)
save(fig, "fig_conformal")

# 3. obfuscation robustness with / without the input guard (REAL smishing texts)
rb = M["robustness_recall_at_alpha_0.02"]
atks = ["clean", "zero-width", "look-alike letters", "leetspeak", "letter spacing", "prompt injection"]
names = ["No attack", "Zero-width\nchars", "Look-alike\nletters", "Leetspeak", "K Y C\nspacing", "Prompt\ninjection"]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
names = [n.replace("\n", " ") for n in names]
for i, a in enumerate(atks):
    lo, hi = rb["raw"][a] * 100, rb["guarded"][a] * 100
    ax.plot([lo, hi], [i, i], color="#CBD2D9", lw=3, zorder=1)
    ax.scatter(lo, i, s=110, color=GRAY, zorder=2, edgecolor="white", linewidth=2, label="Without guard" if i == 0 else None)
    ax.scatter(hi, i, s=110, color=BLUE, zorder=3, edgecolor="white", linewidth=2, label="With ScamTrail guard" if i == 0 else None)
    ax.text(lo - .5, i, f"{lo:.0f}%", ha="right", va="center", fontsize=11, color=INK2)
    ax.text(hi + .5, i, f"{hi:.0f}%", ha="left", va="center", fontsize=11, color=NAVY, fontweight="bold")
ax.set_yticks(range(len(atks))); ax.set_yticklabels(names); ax.invert_yaxis()
ax.set_xlim(82, 103); ax.set_xlabel("Smishing caught at the same 2% false-alert budget (%)")
ax.tick_params(axis="y", length=0)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2, fontsize=11)
fig.suptitle("Scammer tricks vs the input guard (real smishing SMS)", x=0.02, ha="left", y=1.06,
             fontsize=14, color=NAVY, fontweight="bold")
fig.text(0.0, -0.06, "Prompt injection → 100% because a detected injection can only raise risk.", fontsize=9.5, color=INK2)
save(fig, "fig_robustness")
print("ok")

# 4. external Indian validation (never trained on Indian scam messages)
import os
if os.path.exists("results/external.json"):
    X = json.load(open("results/external.json"))["results"]
    conds = [("a_base", "Public\ndata only"), ("b_recalibrate", "+ recali-\nbration"),
             ("c_retrain_recalibrate", "+ bank\nnegatives"), ("d_plus_synthetic_india", "+ India\ntemplates")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, key, title in [(axes[0], "scam_recall_all", "Indian scam probes caught"),
                           (axes[1], "indian_benign_false_alert", "False alerts on Indian SMS traffic")]:
        vals = [X[c][key] * 100 for c, _ in conds]
        cols = [BLUE if c.startswith("d") else GRAY for c, _ in conds]
        bars = ax.bar([l for _, l in conds], vals, color=cols, width=.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * .03, f"{v:.0f}%" if v >= 10 else f"{v:.1f}%",
                    ha="center", fontsize=12, color=NAVY, fontweight="bold")
        if key.endswith("false_alert"):
            ax.axhline(2, color=ORANGE, ls="--", lw=1.5); ax.text(1.0, 4.6, "dashed = 2% budget", color=ORANGE, fontsize=10.5)
        ax.set_title(title, loc="left", fontsize=13, color=NAVY, fontweight="bold")
        ax.tick_params(axis="x", labelsize=10.5); ax.set_ylim(0, max(vals) * 1.25)
    fig.text(0.0, -0.05, "Mean of 10 splits. Probes: 103 Indian scams + 47 legitimate look-alikes (English, Hinglish, Hindi). "
             "Traffic: IIIT-D 2011 + Indian Telecom 2024.", fontsize=9, color=INK2)
    save(fig, "fig_external_india")

# 5. sensitivity sweep
if os.path.exists("results/sensitivity.json"):
    S = json.load(open("results/sensitivity.json"))["grid"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    order2 = ["ScamTrail (full)", "ScamTrail without journey state", "Payment-only model", "RBI rule (new payee > ₹10k)", "Single-SMS classifier"]
    lab2 = ["ScamTrail (full)", "ScamTrail, no journey", "Payment-only", "Blanket rule*", "Single-SMS"]
    for i, (m, l) in enumerate(zip(order2, lab2)):
        v = [r["recall"][m] * 100 for r in S]
        ax.plot([min(v), max(v)], [i, i], color=BLUE if i == 0 else GRAY, lw=8, solid_capstyle="round", alpha=.85)
        ax.text(max(v) + 1.5, i, f"{min(v):.0f}–{max(v):.0f}%", va="center", fontsize=11, color=NAVY, fontweight="bold")
    ax.set_yticks(range(len(lab2))); ax.set_yticklabels(lab2); ax.invert_yaxis(); ax.set_xlim(0, 110)
    ax.set_xlabel("Scam payments caught across 18 simulator settings (%)")
    ax.set_title("Ranking holds when we make the simulator harder", loc="left", fontsize=14, color=NAVY, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    fig.text(0.0, -0.06, "Scam signals ×1 / 0.7 / 0.5, shared messages ×1 / 0.5 / 0, hard negatives ×1 / 2. *Blanket rule at its own 5% budget.",
             fontsize=9, color=INK2)
    save(fig, "fig_sensitivity")
print("ok2")
