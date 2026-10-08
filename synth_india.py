"""Template generator for Indian scam SMS and legitimate look-alikes
(English, Hinglish in Roman script, Hindi in Devanagari).

Why: the public smishing corpus is mostly non-Indian. Indian scams use other
words (UPI PIN, KYC, YONO, "digital arrest", "task", "customs", "bijli") and
code-mixing. Scam typologies follow public RBI / I4C / 1930 advisories:
UPI collect-request reversal, KYC/account block, utility disconnection,
courier/customs + digital arrest, part-time job/task, investment/trading
group, loan app, fake customer care, prize/lottery, family emergency.

Legitimate look-alikes are generated on purpose (real debit alerts, OTPs,
branch KYC reminders, delivery updates) so the model learns the CONTRAST,
not just the topic. Wording is written from the typology, not copied from any
test set. Every number, link and name is random filler.
"""
from __future__ import annotations

import random

import pandas as pd

BANKS = ["SBI", "HDFC", "ICICI", "Axis", "PNB", "Kotak", "Bank of Baroda", "Canara"]
APPS = ["PhonePe", "GPay", "Paytm", "BHIM"]
TELCO = ["Jio", "Airtel", "Vi", "BSNL"]
COURIER = ["India Post", "DTDC", "Blue Dart", "FedEx", "DHL", "Delhivery"]
TLD = ["xyz", "top", "in", "co", "info", "online", "site", "live"]


def _url(r, word):
    return f"{r.choice(['http://', 'https://', ''])}{word}-{r.choice(['verify', 'update', 'kyc', 'help', 'care', 'pay'])}{r.randint(1, 99)}.{r.choice(TLD)}/{r.choice(['a', 'x', 'login', 'p'])}"


def _ph(r):
    return f"{r.choice('6789')}{r.randint(100000000, 999999999)}"


def _amt(r, lo, hi):
    v = r.randint(lo, hi)
    return f"{r.choice(['Rs ', 'Rs.', 'INR ', '₹'])}{v:,}"


# ---------------------------------------------------------------- scams
def scam(r: random.Random, lang: str) -> tuple[str, str]:
    kind = r.choice(["upi_reverse", "kyc_block", "utility", "courier_arrest", "task_job",
                     "invest_group", "loan_app", "fake_care", "prize", "family_emergency"])
    b, a, t, c = r.choice(BANKS), r.choice(APPS), r.choice(TELCO), r.choice(COURIER)
    u, p = _url(r, r.choice([b.lower().replace(" ", ""), a.lower(), t.lower(), "kyc", "refund"])), _ph(r)
    E = {
        "upi_reverse": [f"{a}: {_amt(r, 500, 9999)} cashback is waiting. Approve the collect request and enter your UPI PIN to receive it.",
                        f"I sent you {_amt(r, 1000, 20000)} by mistake on {a}, please accept the request I sent and enter PIN to return it."],
        "kyc_block": [f"Dear {b} customer, your account will be suspended today as KYC is incomplete. Update PAN now: {u}",
                      f"{b} ALERT: your netbanking is blocked. Verify Aadhaar and PAN within 24 hrs at {u} to avoid closure."],
        "utility": [f"Dear consumer, your electricity will be disconnected at 9:30 pm tonight as last month bill is not updated. Contact officer {p}.",
                    f"{t}: your SIM will be deactivated in 2 hours. Complete e-KYC by calling {p} immediately."],
        "courier_arrest": [f"{c}: a parcel in your name has illegal items. Customs and CBI have registered a case. Press 1 or call {p} to avoid arrest.",
                           f"Your {c} package is on hold. Pay {_amt(r, 25, 99)} address fee at {u} within 24 hours or it will be returned."],
        "task_job": [f"Part-time job: earn {_amt(r, 2000, 8000)} daily by liking YouTube videos. Contact HR on WhatsApp {p}.",
                     f"Hello, we are hiring for online tasks. Complete 3 tasks and get {_amt(r, 150, 500)} instantly. Join Telegram group now."],
        "invest_group": [f"Join our VIP stock group, guaranteed {r.randint(20, 300)}% returns this week. Limited seats, register {u}",
                         f"SEBI-approved IPO allotment tip: deposit {_amt(r, 5000, 50000)} today and double it in 7 days."],
        "loan_app": [f"Pre-approved instant loan of {_amt(r, 20000, 500000)} without documents. Pay {_amt(r, 499, 2999)} processing fee at {u}",
                     f"Your loan is sanctioned. To release the amount, pay insurance charge {_amt(r, 999, 4999)} to UPI ID loanhelp{r.randint(1, 99)}@ybl."],
        "fake_care": [f"Your refund of {_amt(r, 500, 5000)} failed. Call {b} customer care {p} and share the code you receive to get it.",
                      f"{a} support: your wallet is restricted. Install AnyDesk and call {p} to unblock."],
        "prize": [f"Congratulations! Your number won {_amt(r, 100000, 2500000)} in the {a} lucky draw. Pay {_amt(r, 999, 9999)} tax to claim at {u}",
                  f"You have won a Tata Safari in KBC lottery. Contact manager {p} to claim the prize."],
        "family_emergency": [f"Hi Papa, this is my new number. My phone is broken and I need {_amt(r, 3000, 30000)} urgently, please send to this UPI.",
                             f"Your son has been detained by police after an accident. Send {_amt(r, 20000, 200000)} immediately to settle the case. Call {p}."],
    }
    H = {
        "upi_reverse": [f"{a} pe aapke liye {_amt(r, 500, 9999)} cashback hai. Paisa lene ke liye request accept karke UPI PIN dalo.",
                        f"Bhai galti se {_amt(r, 1000, 20000)} aapko chale gaye, maine request bheji hai, PIN dal ke wapas kar do please."],
        "kyc_block": [f"Priya grahak, aapka {b} khata aaj band ho jayega kyunki KYC adhura hai. Abhi PAN update karein: {u}",
                      f"{b} account block ho gaya hai. 24 ghante mein Aadhaar verify karo warna khata band: {u}"],
        "utility": [f"Aapki bijli aaj raat 9:30 baje kaat di jayegi, pichla bill update nahi hua. Turant officer ko call karein {p}.",
                    f"{t} SIM 2 ghante mein band ho jayega. e-KYC ke liye abhi call karein {p}."],
        "courier_arrest": [f"Aapke naam ke parcel mein drugs mile hain. CBI ne case darj kiya hai, giraftari se bachne ke liye {p} pe call karein.",
                           f"{c}: aapka parcel ruka hua hai. {_amt(r, 25, 99)} fees bharo is link pe {u} warna wapas chala jayega."],
        "task_job": [f"Ghar baithe roz {_amt(r, 2000, 8000)} kamaye, sirf videos like karne hain. WhatsApp karein {p}.",
                     f"Online task karo aur turant {_amt(r, 150, 500)} pao. Telegram group join karo abhi."],
        "invest_group": [f"Hamare VIP share group mein judiye, is hafte {r.randint(20, 300)}% pakka munafa. Seats kam hain: {u}",
                         f"Aaj {_amt(r, 5000, 50000)} lagao aur 7 din mein double pao, guaranteed."],
        "loan_app": [f"Bina document {_amt(r, 20000, 500000)} ka loan turant. Sirf {_amt(r, 499, 2999)} processing fee bharein: {u}",
                     f"Aapka loan pass ho gaya hai. Paisa release karne ke liye {_amt(r, 999, 4999)} insurance fee is UPI pe bhejein."],
        "fake_care": [f"Aapka refund {_amt(r, 500, 5000)} atak gaya hai. Customer care {p} pe call karke aaya hua code batayein.",
                      f"{a} wallet band hai. AnyDesk app download karke {p} pe call karo, turant chalu ho jayega."],
        "prize": [f"Badhai ho! Aapne {a} lucky draw mein {_amt(r, 100000, 2500000)} jeete hain. Inaam ke liye {_amt(r, 999, 9999)} tax bharein: {u}",
                  f"KBC lottery mein aapki gaadi nikli hai. Inaam lene ke liye manager se baat karein {p}."],
        "family_emergency": [f"Papa ye mera naya number hai, phone toot gaya. Jaldi {_amt(r, 3000, 30000)} bhej do is UPI pe, baad mein batata hu.",
                             f"Aapke bete ko police ne pakda hai. Case khatam karne ke liye turant {_amt(r, 20000, 200000)} bhejein. Call {p}."],
    }
    D = {
        "upi_reverse": [f"{a} पर आपके लिए {_amt(r, 500, 9999)} कैशबैक है। पैसे पाने के लिए रिक्वेस्ट स्वीकार करें और UPI PIN डालें।"],
        "kyc_block": [f"प्रिय ग्राहक, KYC अधूरा होने से आपका {b} खाता आज बंद हो जाएगा। तुरंत अपडेट करें: {u}"],
        "utility": [f"आपकी बिजली आज रात काट दी जाएगी। बिल अपडेट नहीं है, तुरंत अधिकारी को कॉल करें {p}"],
        "courier_arrest": [f"आपके नाम के पार्सल में अवैध सामान मिला है। गिरफ्तारी से बचने के लिए तुरंत {p} पर कॉल करें।"],
        "task_job": [f"घर बैठे रोज़ {_amt(r, 2000, 8000)} कमाएं। सिर्फ वीडियो लाइक करने हैं, अभी व्हाट्सएप करें {p}"],
        "invest_group": [f"हमारे शेयर ग्रुप से जुड़ें, इस हफ्ते {r.randint(20, 300)}% पक्का मुनाफा। अभी रजिस्टर करें {u}"],
        "loan_app": [f"बिना कागज़ {_amt(r, 20000, 500000)} का लोन। सिर्फ {_amt(r, 499, 2999)} प्रोसेसिंग फीस भरें {u}"],
        "fake_care": [f"आपका रिफंड अटका है। कस्टमर केयर {p} पर कॉल करके आया हुआ कोड बताएं।"],
        "prize": [f"बधाई हो! आपने {_amt(r, 100000, 2500000)} का इनाम जीता है। इनाम पाने के लिए {_amt(r, 999, 9999)} टैक्स भरें।"],
        "family_emergency": [f"पापा यह मेरा नया नंबर है, मुझे तुरंत {_amt(r, 3000, 30000)} चाहिए, इस UPI पर भेज दो।"],
    }
    pool = {"en": E, "hinglish": H, "hi": D}[lang][kind]
    return r.choice(pool), kind


# ---------------------------------------------------------------- legitimate look-alikes
def legit(r: random.Random, lang: str) -> tuple[str, str]:
    kind = r.choice(["debit_alert", "otp", "kyc_reminder", "delivery", "bill_paid", "chat_money", "salary"])
    b, a, c = r.choice(BANKS), r.choice(APPS), r.choice(COURIER)
    acc = f"XX{r.randint(1000, 9999)}"
    E = {
        "debit_alert": [f"{_amt(r, 50, 20000)} debited from {b} a/c {acc} on {r.randint(1, 28)}-0{r.randint(1, 9)} to VPA shop{r.randint(1, 99)}@okaxis. Not you? Call the number on your card."],
        "otp": [f"{r.randint(100000, 999999)} is your OTP for {b} netbanking login. Valid for 5 mins. Do not share it with anyone, bank never asks for OTP."],
        "kyc_reminder": [f"Dear customer, periodic KYC update is due. Please visit your nearest {b} branch with ID proof. {b} never asks for PIN, OTP or password."],
        "delivery": [f"Your {c} shipment {r.randint(10**9, 10**10 - 1)} is out for delivery today. Track it in the official app."],
        "bill_paid": [f"Payment of {_amt(r, 300, 5000)} for your electricity bill was successful. Ref {r.randint(10**8, 10**9 - 1)}. Thank you."],
        "chat_money": [f"I have sent the {_amt(r, 200, 3000)} for dinner on {a}, check once.", "Can you send me the rent split by tonight?"],
        "salary": [f"Salary of {_amt(r, 15000, 150000)} credited to your {b} account {acc}. Available balance updated."],
    }
    H = {
        "debit_alert": [f"Bhai {a} pe {_amt(r, 100, 3000)} bhej diye hain, check kar lena."],
        "otp": ["OTP kisi ko mat batana, bank wale kabhi nahi maangte. Main ghar aake baat karta hu."],
        "kyc_reminder": [f"Mummy kal {b} branch jaana hai KYC ke liye, Aadhaar card nikaal ke rakhna."],
        "delivery": ["Parcel aaj aa jayega, guard ko bol dena le le."],
        "bill_paid": ["Bijli ka bill bhar diya maine app se, receipt bhej di hai."],
        "chat_money": [f"Yaar {_amt(r, 200, 3000)} wapas kar diye maine, dekh lena.", "Kal party ka paisa sab baraabar baant lenge."],
        "salary": ["Salary aa gayi hai, shaam ko market chalte hain."],
    }
    D = {
        "debit_alert": [f"आपके {b} खाते {acc} से {_amt(r, 100, 5000)} का भुगतान हुआ है। आप नहीं हैं तो शाखा से संपर्क करें।"],
        "otp": ["OTP किसी को न बताएं। बैंक कभी OTP नहीं मांगता।"],
        "kyc_reminder": [f"KYC अपडेट के लिए अपनी नज़दीकी {b} शाखा में जाएं।"],
        "delivery": ["आपका पार्सल आज पहुंच जाएगा।"],
        "bill_paid": ["बिजली बिल का भुगतान सफल रहा। धन्यवाद।"],
        "chat_money": ["मैंने पैसे भेज दिए हैं, देख लेना।"],
        "salary": ["आपके खाते में वेतन जमा हो गया है।"],
    }
    pool = {"en": E, "hinglish": H, "hi": D}[lang][kind]
    return r.choice(pool), kind


def generate(n_scam=3000, n_legit=3000, seed=0) -> pd.DataFrame:
    r = random.Random(seed)
    langs, w = ["en", "hinglish", "hi"], [0.45, 0.4, 0.15]
    rows = []
    for _ in range(n_scam):
        lang = r.choices(langs, w)[0]
        t, k = scam(r, lang)
        rows.append((t, 1, k, lang))
    for _ in range(n_legit):
        lang = r.choices(langs, w)[0]
        t, k = legit(r, lang)
        rows.append((t, 0, k, lang))
    d = pd.DataFrame(rows, columns=["text", "is_smish", "kind", "lang"]).drop_duplicates("text")
    return d.reset_index(drop=True)


if __name__ == "__main__":
    g = generate()
    print(g.groupby(["lang", "is_smish"]).size().unstack())
    print(g.sample(6, random_state=1).to_string())
