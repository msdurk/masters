# components.py
# Shared components for email classifier

import re
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

class TextStatsTransformer(BaseEstimator, TransformerMixin):
    """
    Hand-crafted numeric features from raw text.
    Keep this in a shared module so pickled models unpickle reliably.
    """
    URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
    HTML_TAG_RE = re.compile(r"<[^>]+>")
    SUSPICIOUS_TLDS = (".ru",".cn",".top",".xyz",".tk",".icu",".click",".link",".work",".country")
    URGENT_WORDS = (
        "urgent","verify","suspend","suspended","login","immediately","click",
        "password","account","confirm","update","security","expire","expired","invoice","payment"
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for text in X:
            t = str(text)
            length = max(len(t), 1)

            urls = self.URL_RE.findall(t)
            url_count = len(urls)

            html_tags = len(self.HTML_TAG_RE.findall(t))

            suspicious = sum(1 for u in urls
                             if any(u.lower().endswith(tld) or tld in u.lower()
                                    for tld in self.SUSPICIOUS_TLDS))

            exclam = t.count("!")

            letters = [ch for ch in t if ch.isalpha()]
            upper = sum(1 for ch in letters if ch.isupper())
            upper_ratio = upper / max(len(letters), 1)

            digits = sum(ch.isdigit() for ch in t)
            digit_ratio = digits / length

            money = t.count("$") + t.count("€") + t.count("£")
            at_count = t.count("@")
            urgent_hits = sum(t.lower().count(w) for w in self.URGENT_WORDS)

            feats.append([
                url_count, html_tags, suspicious, exclam,
                upper_ratio, digit_ratio, money, at_count, urgent_hits, length
            ])

        mat = np.asarray(feats, dtype=float)
        # log-scale counts; keep ratios as-is
        mat[:, 0] = np.log1p(mat[:, 0])   # url_count
        mat[:, 1] = np.log1p(mat[:, 1])   # html_tags
        mat[:, 2] = np.log1p(mat[:, 2])   # suspicious
        mat[:, 3] = np.log1p(mat[:, 3])   # exclam
        mat[:, 6] = np.log1p(mat[:, 6])   # money
        mat[:, 7] = np.log1p(mat[:, 7])   # at_count
        mat[:, 8] = np.log1p(mat[:, 8])   # urgent_hits
        mat[:, 9] = np.log1p(mat[:, 9])   # length
        return sparse.csr_matrix(mat)
