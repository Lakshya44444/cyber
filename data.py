"""Load and clean the public SMS phishing dataset (Mishra & Soni, 2022).

Labels: ham / spam / smishing. Many SMS datasets contain repeated templates
(same message with a different number or prize amount). If copies land in
both train and test, scores look better than they are. We therefore:
  1. drop exact duplicates after normalisation, and
  2. build a "template id" (digits, URLs and amounts masked) so that
     cross-validation never puts two copies of one template on both sides.
"""
import re
import pandas as pd

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|in|co|net|org|ly|me)\S*)", re.I)
NUM_RE = re.compile(r"\d+")


def normalise_for_template(text: str) -> str:
    t = text.lower()
    t = URL_RE.sub(" <url> ", t)
    t = NUM_RE.sub("#", t)
    t = re.sub(r"[^a-z#<> ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_smishing(path: str = "data/Dataset_5971.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df = df.rename(columns=str.lower)
    df["label"] = df["label"].str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.replace("\t", " ").str.strip()
    df["text"] = df["text"].str.replace("ï¿½", "'", regex=False)
    df["template"] = df["text"].map(normalise_for_template)
    n_raw = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    n_exact = len(df)
    df["group"] = df["template"].factorize()[0]
    df["is_smish"] = (df["label"] == "smishing").astype(int)
    df.attrs["stats"] = {
        "raw_rows": n_raw,
        "after_exact_dedup": n_exact,
        "templates": int(df["group"].nunique()),
        "label_counts": df["label"].value_counts().to_dict(),
    }
    return df


if __name__ == "__main__":
    d = load_smishing()
    print(d.attrs["stats"])
