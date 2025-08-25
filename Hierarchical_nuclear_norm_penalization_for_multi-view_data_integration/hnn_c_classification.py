# hnn_c_language_classifier_progress.py
# Usage:
#   python hnn_c_language_classifier_progress.py /path/to/dataset.json

import json, sys, re, numpy as np
from pathlib import Path

from tqdm import tqdm

# ---- Import base HNN (from the earlier implementation you saved as hnn.py)
from hnn_impl import HNN, HNNOperators, column_stack_views, split_columns, frob2, nuclear_norm
import numpy as np
from numpy.linalg import svd

# --------- A thin subclass that shows a tqdm bar during HNN.fit ----------
class HNNProgress(HNN):
    def fit(self, X_list):
        # (This is the same as HNN.fit, with a tqdm bar wrapped around the main loop)
        X_list_proc, _scales = self._preprocess_views(X_list)
        n = X_list_proc[0].shape[0]
        assert all(X.shape[0] == n for X in X_list_proc), "All views must have matching sample size n."
        p_list = [X.shape[1] for X in X_list_proc]
        ops = HNNOperators(p_list)
        self._init_weights(ops)

        X_all = column_stack_views(X_list_proc)
        M_all = np.zeros_like(X_all)
        M_prev = M_all.copy()
        Y = [np.zeros((n, len(op['cols']))) for op in ops.ops]

        # Map group -> lambda weight per operator
        group_lams = []
        ind_idx = 0
        pair_idx = 0
        for op in ops.ops:
            if op['group'] == 'individual':
                lam = self.tau * float(self.w_ind[ind_idx])
                ind_idx += 1
            elif op['group'] == 'pairwise':
                lam = self.kappa * float(self.w_pair[pair_idx])
                pair_idx += 1
            else:
                lam = self.lam_all
            group_lams.append(lam)

        self.history_ = {'primal_res': [], 'obj': []}

        with tqdm(total=self.max_iter, desc="HNN fitting (iterations)", unit="it") as pbar:
            for it in range(self.max_iter):
                # Dual updates + prox of g*
                for j, op in enumerate(ops.ops):
                    cols = op['cols']
                    Y[j] = Y[j] + self.s_d * M_all[:, cols]
                    # prox of conjugate (via Moreau)
                    lam = group_lams[j]
                    if lam > 0:
                        Y_j = Y[j]
                        # prox_{sigma g*}(Y) = Y - sigma * SVT(Y/sigma, lam/sigma)
                        U, s, Vt = svd(Y_j / self.s_d, full_matrices=False)
                        s_thr = np.maximum(s - lam / self.s_d, 0.0)
                        r = np.sum(s_thr > 0)
                        if r == 0:
                            Y[j] = np.zeros_like(Y_j)
                        else:
                            Y[j] = Y_j - self.s_d * ((U[:, :r] * s_thr[:r]) @ Vt[:r, :])

                # Primal update
                Z = M_all.copy()
                A = np.zeros_like(M_all)
                for j, op in enumerate(ops.ops):
                    cols = op['cols']
                    A[:, cols] += Y[j]
                Z = Z - self.t_p * A
                M_all = (Z + self.t_p * X_all) / (1.0 + self.t_p)

                primal_res = np.linalg.norm(M_all - M_prev, 'fro') / (np.linalg.norm(M_prev, 'fro') + 1e-12)
                M_prev = M_all.copy()

                # Objective (monitor)
                obj = 0.5 * frob2(X_all - M_all)
                # individuals
                for d in range(ops.D):
                    cols = ops.level3_cols[d]
                    U_sing = svd(M_all[:, cols], full_matrices=False, compute_uv=False)
                    obj += self.tau * float(self.w_ind[d]) * float(np.sum(U_sing))
                # pairwise
                for j, pair in enumerate(ops.level2_cols):
                    U_sing = svd(M_all[:, pair], full_matrices=False, compute_uv=False)
                    obj += self.kappa * float(self.w_pair[j]) * float(np.sum(U_sing))
                # all
                U_sing = svd(M_all, full_matrices=False, compute_uv=False)
                obj += self.lam_all * float(np.sum(U_sing))

                self.history_['primal_res'].append(primal_res)
                self.history_['obj'].append(obj)

                pbar.set_postfix({"res": f"{primal_res:.2e}", "obj": f"{obj:.3e}"})
                pbar.update(1)

                if primal_res < self.tol:
                    break

        col_slices = [slice(ops.offsets[d], ops.offsets[d] + p_list[d]) for d in range(ops.D)]
        self.M_list_ = split_columns(M_all, col_slices)

        # Refitting (with its own small bar)
        if self.refit:
            self.M_refit_ = []
            for X, M in tqdm(list(zip(X_list_proc, self.M_list_)), desc="Refit per view", unit="view"):
                if np.allclose(M, 0):
                    self.M_refit_.append(M.copy())
                    continue
                U, s, Vt = svd(M, full_matrices=False)
                r = np.sum(s > 1e-10)
                if r == 0:
                    self.M_refit_.append(np.zeros_like(M))
                else:
                    U_r = U[:, :r]
                    M_refit = U_r @ (U_r.T @ X)
                    self.M_refit_.append(M_refit)
        else:
            self.M_refit_ = None
        return self

# ---------------- Base pipeline with progress bars ----------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

C_HEUR_PATTERNS = [
    r"#\s*include\s*<[^>]+>",
    r"\bscanf\s*\(",
    r"\bprintf\s*\(",
    r"\bmalloc\s*\(",
    r"\bfree\s*\(",
    r"\bint\s+main\s*\(",
    r"\btypedef\b",
    r"\bstruct\b",
    r"\bNULL\b",
]
def weak_is_c(code: str) -> int:
    if not code or not isinstance(code, str):
        return 0
    hits = sum(bool(re.search(p, code)) for p in C_HEUR_PATTERNS)
    return 1 if hits >= 2 else 0

def load_data(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    X_code, X_prompt, X_meta, y = [], [], [], []
    for row in tqdm(data, desc="Loading rows", unit="row"):
        code = row.get("c_code", "") or ""
        prompt = row.get("prompt", "") or ""
        X_code.append(code)
        X_prompt.append(prompt)
        meta_feats = [
            float(row.get("char_count", 0.0)),
            float(row.get("num_lines", 0.0)),
            float(row.get("nloc", 0.0)),
            float(row.get("CC", 0.0)),
            float(row.get("token_size", 0.0)),
        ]
        model_name = str(row.get("model_name", ""))
        bucket = (hash(model_name) % 10) / 10.0
        meta_feats.append(bucket)
        X_meta.append(meta_feats)
        if "is_c" in row:
            y.append(int(row["is_c"]))
        elif "language" in row:
            y.append(1 if str(row["language"]).strip().lower() in {"c", "c99", "c11"} else 0)
        else:
            y.append(weak_is_c(code))
    return X_code, X_prompt, np.asarray(X_meta, dtype=float), np.asarray(y, dtype=int)

def main():
    if len(sys.argv) < 2:
        print("Usage: python hnn_c_language_classifier_progress.py /path/to/dataset.json")
        sys.exit(1)
    path = Path(sys.argv[1])

    print("Step 1/6: Load data")
    X_code_raw, X_prompt_raw, X_meta_raw, y = load_data(path)

    print("Step 2/6: Build views (TF-IDF + scaling)")
    with tqdm(total=3, desc="Vectorizers & scaler", unit="step") as p:
        code_vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_features=20000)
        X_code = code_vec.fit_transform(X_code_raw).astype(np.float32).toarray()
        p.update(1)

        prompt_vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, max_features=10000)
        X_prompt = prompt_vec.fit_transform(X_prompt_raw).astype(np.float32).toarray()
        p.update(1)

        scaler = StandardScaler()
        X_meta = scaler.fit_transform(X_meta_raw.astype(np.float32))
        p.update(1)

    n = len(y)
    assert X_code.shape[0] == X_prompt.shape[0] == X_meta.shape[0] == n

    print("Step 3/6: Fit HNN (with progress)")
    hnn = HNNProgress(
        tau_individual=0.6,
        kappa_pairwise=0.4,
        lam_all=0.3,
        step_primal=0.2,
        step_dual=0.2,
        max_iter=1000,
        tol=1e-5,
        refit=True,
        random_state=0,
    )
    hnn.fit([X_code, X_prompt, X_meta])

    print("Step 4/6: Get fused representation")
    M_code, M_prompt, M_meta = hnn.get_signal(refit=True)
    Z = np.concatenate([M_code, M_prompt, M_meta], axis=1)

    print("Step 5/6: Train/evaluate classifier (5-fold CV)")
    from sklearn.utils import check_random_state
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_pred_all, y_prob_all = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(list(skf.split(Z, y)), desc="CV folds", unit="fold"), 1):
        clf = LogisticRegression(max_iter=200)
        clf.fit(Z[train_idx], y[train_idx])
        y_prob = clf.predict_proba(Z[test_idx])[:,1]
        y_pred = (y_prob >= 0.5).astype(int)
        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(y_pred.tolist())
        y_prob_all.extend(y_prob.tolist())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)

    print("\n=== HNN-fused C vs not-C classification ===")
    print(classification_report(y_true_all, y_pred_all, digits=3))
    try:
        print("ROC AUC:", roc_auc_score(y_true_all, y_prob_all))
    except Exception:
        pass
    print("Accuracy:", accuracy_score(y_true_all, y_pred_all))

    print("Step 6/6: Demo inference (with progress)")
    demo_records = [
        {
            "model_name": "demo",
            "prompt": "Write a C program that prints the first 10 Fibonacci numbers.",
            "c_code": "#include <stdio.h>\nint main(){int a=0,b=1,c;for(int i=0;i<10;i++){printf(\"%d \",a);c=a+b;a=b;b=c;}return 0;}",
            "char_count": 180, "num_lines": 4, "nloc": 3, "CC": 2, "token_size": 60
        },
        {
            "model_name": "demo",
            "prompt": "Write the same in Python.",
            "c_code": "def fib(n):\n a,b=0,1\n for _ in range(n):\n  print(a)\n  a,b=b,a+b",
            "char_count": 95, "num_lines": 5, "nloc": 4, "CC": 2, "token_size": 35
        },
        {
            "model_name": "demo",
            "prompt": "Make a Java method for factorial",
            "c_code": "public static int fact(int n){return n<=1?1:n*fact(n-1);}",
            "char_count": 72, "num_lines": 1, "nloc": 1, "CC": 1, "token_size": 20
        }
    ]

    # Build demo features
    code_txts = [r.get("c_code","") for r in demo_records]
    prompt_txts = [r.get("prompt","") for r in demo_records]
    metas = []
    for r in demo_records:
        meta = [
            float(r.get("char_count", 0.0)),
            float(r.get("num_lines", 0.0)),
            float(r.get("nloc", 0.0)),
            float(r.get("CC", 0.0)),
            float(r.get("token_size", 0.0)),
        ]
        bucket = (hash(str(r.get("model_name",""))) % 10) / 10.0
        meta.append(bucket)
        metas.append(meta)
    Xc = code_vec.transform(code_txts).astype(np.float32).toarray()
    Xp = prompt_vec.transform(prompt_txts).astype(np.float32).toarray()
    metas = scaler.transform(np.asarray(metas, dtype=float))

    # Project with learned subspaces (quick approximation)
    def project_with_UU_T(M_train, X_new):
        if np.allclose(M_train, 0):
            return np.zeros_like(X_new)
        U, s, _ = np.linalg.svd(M_train, full_matrices=False)
        r = (s > 1e-10).sum()
        if r == 0: return np.zeros_like(X_new)
        U = U[:, :r]
        return U @ (U.T @ X_new)

    # Reuse the last trained classifier from CV for a simple demo
    with tqdm(total=3, desc="Projecting demo views", unit="view") as pb:
        Z_demo = np.concatenate([
            project_with_UU_T(M_code, Xc),
            project_with_UU_T(M_prompt, Xp),
            project_with_UU_T(M_meta, metas),
        ], axis=1)
        pb.update(3)

    # Train a quick final classifier on all data for demo prediction
    clf_final = LogisticRegression(max_iter=200).fit(Z, y)
    probs = clf_final.predict_proba(Z_demo)[:,1]
    print("\nDemo probabilities (is C):", probs)

if __name__ == "__main__":
    main()
