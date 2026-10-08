"""Message-intent model (the on-device "small model" in the architecture).

In the hackathon prototype this is a TF-IDF + linear model: tiny, fast, and
explainable. The LLM extractor in the design is a drop-in upgrade behind the
same schema-locked interface (guardrails.Intent).
"""
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from guardrails import detect_injection, guard_message

INJECTION_FLOOR = 0.90  # an attempt to talk to the model is itself evidence


class TextFeatures:
    def __init__(self):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                    sublinear_tf=True, max_features=60000)

    def fit_transform(self, texts):
        return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()

    def transform(self, texts):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()


def make_clf(kind: str):
    if kind == "logreg":
        return LogisticRegression(max_iter=3000, C=8.0, class_weight="balanced")
    if kind == "linear_svm":
        return CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3)
    if kind == "naive_bayes":
        return ComplementNB(alpha=0.3)
    raise ValueError(kind)


class MessageModel:
    """Binary smishing scorer: P(smishing | text)."""

    def __init__(self, kind="logreg", guarded=True):
        self.kind, self.guarded = kind, guarded
        self.feats, self.clf = TextFeatures(), make_clf(kind)

    def _prep(self, texts):
        if not self.guarded:
            return list(texts), np.zeros(len(texts), bool)
        g = [guard_message(t) for t in texts]
        return [x["text"] for x in g], np.array([bool(x["injection"]) for x in g])

    def fit(self, texts, y):
        X = self.feats.fit_transform(self._prep(texts)[0])
        self.clf.fit(X, y)
        return self

    def score(self, texts):
        clean, inj = self._prep(texts)
        p = self.clf.predict_proba(self.feats.transform(clean))[:, 1]
        # output guard: a detected injection can only raise risk
        return np.where(inj, np.maximum(p, INJECTION_FLOOR), p)
