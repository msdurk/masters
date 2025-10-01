
#!/usr/bin/env python3
# Option 3A: LLM embeddings (+ engineered features) -> small classifier
# Usage:
#   python pro_train_embed.py --outdir models/embed_v1
# Flags allow toggling embedding backends:
#   --embed-backend sentence-transformers --model all-MiniLM-L6-v2
#   --embed-backend openai --model text-embedding-3-small
#   --embed-backend tfidf   (fallback, no internet required)
#
# The script:
#   1) Loads your provided CSVs
#   2) Maps labels to wanted/unwanted (1=unwanted, 0=wanted)
#   3) Builds embeddings (or TF-IDF if selected)
#   4) Adds engineered features (URL count, length, uppercase ratio)
#   5) Trains LogisticRegression (class_weight='balanced')
#   6) Saves artifacts to outdir
import os, argparse, json, sys, re
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Local utils
from _loader_utils import read_simple_csv, read_problem_llm_phishing, unify_frames
from components import TextStatsTransformer

def _to_2d(X):
    """Ensure array/matrix is 2D with shape (n_samples, n_features)."""
    if sparse.issparse(X):
        # already 2D if sparse; if someone passed a 1D dense array, convert below
        return X
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)   # (n_samples,) -> (n_samples, 1)
    return X

def _hstack(A, B):
    """Horizontally stack dense/sparse safely."""
    A = _to_2d(A)
    B = _to_2d(B)
    assert A.shape[0] == B.shape[0], f"Row mismatch: {A.shape} vs {B.shape}"
    if sparse.issparse(A) or sparse.issparse(B):
        # make both sparse for consistent stacking
        if not sparse.issparse(A): A = sparse.csr_matrix(A)
        if not sparse.issparse(B): B = sparse.csr_matrix(B)
        return sparse.hstack([A, B], format="csr")
    else:
        return np.hstack([A, B])


def engineered_features(df: pd.DataFrame) -> np.ndarray:
    texts = df["text"].fillna("").astype(str).values
    feats = []
    url_pattern = re.compile(r'https?://|www\\.')
    for t in texts:
        n_chars = len(t)
        n_urls = len(url_pattern.findall(t))
        n_caps = sum(1 for ch in t if ch.isupper())
        cap_ratio = (n_caps / (n_chars+1e-6))
        feats.append([n_chars, n_urls, cap_ratio])
    return np.array(feats, dtype=float)

def build_embedder(args):
    if args.embed_backend == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(args.model)
            def embed(texts: list[str]) -> np.ndarray:
                return np.array(model.encode(texts, show_progress_bar=False))
            return embed, {"backend":"sentence-transformers", "model": args.model}
        except Exception as e:
            print(f"[WARN] sentence-transformers failed: {e}. Falling back to TF-IDF.", file=sys.stderr)
            args.embed_backend = "tfidf"

    if args.embed_backend == "openai":
        try:
            import openai  # requires OPENAI_API_KEY in env
            client = openai.OpenAI()
            model = args.model
            def embed(texts: list[str]) -> np.ndarray:
                out = []
                # Chunk to avoid large requests
                for i in range(0, len(texts), args.batch_size):
                    batch = texts[i:i+args.batch_size]
                    resp = client.embeddings.create(model=model, input=batch)
                    out.extend([d.embedding for d in resp.data])
                return np.array(out, dtype=float)
            return embed, {"backend":"openai", "model": model}
        except Exception as e:
            print(f"[WARN] OpenAI embeddings failed: {e}. Falling back to TF-IDF.", file=sys.stderr)
            args.embed_backend = "tfidf"

    if args.embed_backend == "tfidf":
        # We'll fit inside the pipeline; return placeholder
        def embed(texts: list[str]) -> np.ndarray:
            raise RuntimeError("TF-IDF embed is handled inside sklearn pipeline.")
        return embed, {"backend":"tfidf", "model":"char-3-5"}

    raise ValueError("Unknown embed backend")

def load_all_data() -> pd.DataFrame:
    # Provided files
    base = "/fp/homes01/u01/ec-mathiassd/phishing_detection/data/test_case/"
    frames = []
    # HuggingFace-like corpora
    frames.append(read_simple_csv(os.path.join(base, "hug_legit.csv")).assign(source="hug_legit", assumed_label=0))
    frames.append(read_simple_csv(os.path.join(base, "hug_phishing.csv")).assign(source="hug_phishing", assumed_label=1))
    # Human datasets
    human_legit = read_simple_csv(os.path.join(base, "human_legit.csv")).assign(source="human_legit")
    human_phish = read_simple_csv(os.path.join(base, "humen_phishing.csv")).assign(source="humen_phishing")
    # Their labels appear in 'label' already (1), but ensure unwanted=1, wanted=0
    human_legit["label"] = 0
    human_phish["label"] = 1
    frames += [human_legit[["sender","receiver","date","subject","body","urls","label"]].rename(columns={"body":"text"})[["text","label"]],
               human_phish[["sender","receiver","date","subject","body","urls","label"]].rename(columns={"body":"text"})[["text","label"]]]
    # LLM synthetic
    llm_legit = read_simple_csv(os.path.join(base, "llm_legit.csv")).assign(source="llm_legit")
    llm_legit["label"] = 0
    llm_phish = read_simple_csv(os.path.join(base, "llm_phishing_fixed.csv")).assign(source="llm_phishing")
    llm_phish["label"] = 1
    # Hug sets have only text. Use assumed_label
    frames[0]["label"] = 0
    frames[1]["label"] = 1

    df = pd.concat([f[["text","label"]] for f in frames], ignore_index=True)
    # Clean minor whitespace
    df["text"] = df["text"].astype(str).str.replace(r"\\s+", " ", regex=True).str.strip()
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="models/embed_v1")
    ap.add_argument("--embed-backend", choices=["sentence-transformers","openai","tfidf"], default="tfidf")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_all_data()
    y = df["label"].astype(int).values

    # Split (stratified random; swap to time-based if you add timestamps consistently)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=42)

    embed_fn, embed_meta = build_embedder(args)

    # Feature blocks
    if args.embed_backend == "tfidf":
        text_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=3, max_features=300_000)
        from sklearn.base import BaseEstimator, TransformerMixin
        class EngineeredFeats(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None): return self
            def transform(self, X):
                return engineered_features(X if isinstance(X, pd.DataFrame) else pd.DataFrame({"text":X}))

        feats = ColumnTransformer([
            ("tfidf", text_vectorizer, "text"),
            ("stats", TextStatsTransformer(), "text"),
        ])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
        pipe = Pipeline([("feats", feats), ("clf", clf)])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:,1]
        ap = average_precision_score(y_test, probs)
        print("PR-AUC (unwanted):", round(ap,4))
        print(classification_report(y_test, (probs>=0.5).astype(int), target_names=["wanted","unwanted"]))
        joblib.dump(pipe, os.path.join(args.outdir, "model.joblib"))
        joblib.dump({"type":"sklearn_tfidf+engineered"}, os.path.join(args.outdir, "meta.json"))
    else:
        # Precompute embeddings then combine with engineered features
        texts_train = X_train["text"].tolist()
        texts_test  = X_test["text"].tolist()
        E_train = embed_fn(texts_train)
        E_test  = embed_fn(texts_test)
        stats = TextStatsTransformer()
        F_train = stats.fit_transform(X_train['text'])
        F_test  = stats.transform(X_test['text'])
        Xtr = _hstack(E_train, F_train)
        Xte = _hstack(E_test, F_test)

        scaler = StandardScaler(with_mean=False)  # for sparse compatibility; (embeddings are dense)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
        # We use a simple Pipeline: scale -> clf
        pipe = Pipeline([("scaler", scaler), ("clf", clf)])
        pipe.fit(Xtr, y_train)
        probs = pipe.predict_proba(Xte)[:,1]
        ap = average_precision_score(y_test, probs)
        print("PR-AUC (unwanted):", round(ap,4))
        print(classification_report(y_test, (probs>=0.5).astype(int), target_names=["wanted","unwanted"]))

        # Save artifacts
        joblib.dump({"scaler": pipe.named_steps["scaler"], "clf": pipe.named_steps["clf"]}, os.path.join(args.outdir, "model.joblib"))
        joblib.dump({"type":"embed+engineered", "embed_meta": embed_meta}, os.path.join(args.outdir, "meta.json"))
        # Save train-time engineered feature spec (just to be explicit)
        joblib.dump({"engineered":["n_chars","n_urls","cap_ratio"]}, os.path.join(args.outdir, "engineered_spec.json"))

if __name__ == "__main__":
    main()
