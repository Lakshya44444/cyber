# ScamTrail

**We detect the scam journey, not the scam message.**

ScamTrail is a prototype for UPI scam detection. It runs on the phone, needs the user's consent, and tracks the whole scam journey (contact → hook → pressure → unusual action → payment). It steps in at the UPI PIN screen with graded friction (T0–T3). Guardrails sit around every model input and output.

Built for the Amazon AI Cyber Security Hackathon, Track 02 (AI-Driven Scam Pattern Recognition).

## Architecture

```mermaid
flowchart LR
    A[1 Consented signals<br/>shared SMS, call yes/no,<br/>device flags, payment] --> B[2 Input guardrails<br/>de-obfuscate, rules + learned<br/>injection detector, redact PII]
    B --> C[3 Message extractor<br/>EN / Hinglish / Hindi,<br/>schema-locked, can only raise risk]
    C --> D[4 Journey tracker<br/>stage scores S0–S3, time decay]
    A --> D
    D --> E[5 Risk engine<br/>gradient boosting + conformal tiers;<br/>a hold needs two signal families]
    E --> F[6 PIN-screen intervention<br/>T0–T3 + output guard on explanation]
    F -->|Tier 3| G[Bank analyst / trusted person / 1930]
```

| Guardrail | Where | What it does |
|---|---|---|
| `normalise()` | input | undoes zero-width chars, look-alike letters, leetspeak and "K Y C" spacing; protects real UPI IDs, emails and PAN from being altered |
| `detect_injection()` + `injection_model.py` | input | regex rules and a character n-gram detector find text aimed at the model and remove it; a detected injection fixes risk at ≥ 0.9 |
| `redact_pii()` | input | masks phone, UPI, Aadhaar, PAN, card, account, OTP and email; keeps only the domain of a link |
| `Intent` (pydantic) | output | the extractor must return this schema, and anything else is rejected (never read as "safe") |
| `monotonic()` | output | a shared message can only raise a stage score |
| `check_explanation()` | output | blocks "safe", accusations, threats, score leaks and reasons the model did not use; falls back to a fixed template |

## Data

| Data | Role | Size |
|---|---|---|
| Mishra & Soni (2022), *SMS Phishing Dataset*, Mendeley Data ([f45bkkt8pr](https://data.mendeley.com/datasets/f45bkkt8pr)) | training and real-SMS results | 5,831 unique messages |
| `synth_india.py`: India scam and look-alike templates (ours) written from RBI / I4C typologies | training | about 6,000 messages (English, Hinglish, Hindi) |
| IIIT-Delhi Indian SMS (Yadav, Kumaraguru et al., 2011) | **external test only** | 1,982 messages |
| Indian Telecom SMS Spam Collection (2024, MIT) | **external test only** | 2,042 messages |
| spam-scan Indian scam probes (2026, MIT) | **external test only** | 103 scams + 47 legitimate look-alikes |
| `journey_sim.py`: simulated journeys (ours) | journey benchmark | 16,000 users, 161k payments |

Run `./fetch_external.sh` to download the external sets. Splits are grouped by message template, so near-copies never sit in both train and test. No bank, customer or platform data is used.

## Results

### 1. Real SMS (Mishra & Soni)
| | Value |
|---|---|
| PR-AUC (grouped 5-fold) | 0.95 |
| Measured false alerts at 1% / 2% / 5% targets (20 splits) | 1.0% / 2.0% / 5.4% |
| Smishing caught at a 2% budget | 92% |
| Recall under obfuscation, without guard → with guard | 87–89% → 92–93% |
| PII redaction through the full guard chain (8 types) | 100% |

### 2. Indian external test (no Indian scam message is used for training)
![external](results/fig_external_india.png)

- A model trained on public data alone catches **30%** of Indian scam probes. On Indian SMS traffic its false alerts reach **8.3%**, more than four times the 2% budget.
- Re-setting the threshold on bank traffic (which needs no scam labels) restores the budget, but recall falls to 14%.
- Adding the India typology templates brings recall to **71%** at **2.0%** false alerts: Hinglish ~100%, Hindi 100% (n = 5), English 68%, misspelled 33%. Recall on the original SMS test stays at 87%.
- *Caveat:* we read the probe set while building the templates, so this is a development test, not a blind one. The blind test is bank shadow mode.

### 3. Journey benchmark (same 2% false-alert budget for every method)
| Method | Scam payments caught |
|---|---|
| Blanket rule (new payee > ₹10k, its own 5% budget) | 33% |
| Single-SMS classifier | 8% |
| Payment-only model | 53% |
| ScamTrail without journey state | 76% |
| **ScamTrail (full)** | **95%** |

**Tiered friction:**
- A Tier-2 pause catches 95% of scam payments, with prompts on 2.2% of genuine payments.
- Tier-3 holds need two independent signal families. They hold 52% of scam payments and reach 88% of scam journeys, while holding only 0.2% of genuine payments.

### 4. Is the ranking an artefact of the simulator?
![sensitivity](results/fig_sensitivity.png)

We re-ran the benchmark in 18 settings: scam signals at ×1, 0.7 and 0.5; shared messages at ×1, 0.5 and 0; hard negatives at ×1 and 2.
- ScamTrail ranks first in **18 of 18** settings, catching 81–95% of scam payments against 45–55% for payment-only.
- One assumption is not varied: the repeated, growing payments of task scams, which I4C advisories document.

### 5. Prompt injection the rules have never seen
| | Detected |
|---|---|
| Regex rules alone | 17% |
| Regex + learned detector | 45% |

This is tested leave-one-family-out over five injection styles. False triggers on real SMS sentences: 0.1%. Detection is still partial, which is why the LLM never decides: its output is schema-locked and can only raise risk.

## Run it

```bash
pip install -r requirements.txt
./fetch_external.sh          # external Indian test sets
python run_all.py            # every experiment + figures (~12 min)
pytest -q                    # guardrail unit tests
```

## Files

| File | Purpose |
|---|---|
| `guardrails.py` | all input and output guardrails |
| `injection_model.py` | learned injection detector, leave-one-family-out test |
| `message_model.py` | TF-IDF word + character model behind the input guard |
| `synth_india.py` | India scam / look-alike template generator |
| `external.py`, `run_external.py` | Indian external test sets and transfer / adaptation experiment |
| `conformal.py` | split-conformal and Mondrian thresholds |
| `journey_sim.py`, `tracker.py`, `run_journey.py` | journey simulator, kill-chain tracker, benchmark |
| `run_sensitivity.py` | 18-setting stress test of the simulator |
| `attacks.py` | red-team obfuscation and injection attacks (evaluation only) |
| `tests/` | guardrail unit tests (run in GitHub Actions) |

## Limitations

- Journey results come from a simulator. Real-world performance has to be checked in bank shadow mode.
- The Indian probe set is small (150 messages) and was seen during development. Recall on misspelled scams is low (33%).
- Learned injection detection is partial (45% on unseen styles).
- Per-age and per-language (Mondrian) calibration gave no gain in simulation.
- The prototype uses histogram gradient boosting and occlusion attribution in place of XGBoost and SHAP.
