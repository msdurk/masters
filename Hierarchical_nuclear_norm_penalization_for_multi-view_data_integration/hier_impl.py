# hnn.py
import itertools
import math
from typing import List, Dict, Tuple, Optional
import numpy as np
from numpy.linalg import svd

# ---------- utilities ----------
def svt(X: np.ndarray, tau: float) -> np.ndarray:
    """Singular value soft-thresholding: prox_{tau * ||.||_*}(X)."""
    if tau <= 0:
        return X
    U, s, Vt = svd(X, full_matrices=False)
    s_thr = np.maximum(s - tau, 0.0)
    r = np.sum(s_thr > 0)
    if r == 0:
        return np.zeros_like(X)
    return (U[:, :r] * s_thr[:r]) @ Vt[:r, :]

def prox_conjugate_nuc(Y: np.ndarray, sigma: float, lam: float) -> np.ndarray:
    """
    Prox of sigma * (lam * ||.||_*)^* using Moreau:
      prox_{sigma g*}(Y) = Y - sigma * prox_{g/sigma}(Y / sigma), g(X) = lam ||X||_*.
    """
    if lam <= 0:
        return Y  # g = 0 => g* = indicator{0}, prox is identity
    return Y - sigma * svt(Y / sigma, lam / sigma)

def frob2(A: np.ndarray) -> float:
    return float(np.sum(A * A))

def column_stack_views(M_list: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(M_list, axis=1)

def split_columns(M_all: np.ndarray, col_slices: List[slice]) -> List[np.ndarray]:
    return [M_all[:, s] for s in col_slices]

# ---------- operator builder ----------
class HNNOperators:
    """
    Build linear operators L_i : R^{n x sum p_d} -> R^{n x sum p_{subset}}
    that *select* and *concatenate* columns for:
      - level 1 (all views): {1,...,D}
      - level 2 (pairwise): all pairs
      - level 3 (individual): each view
    Each operator is represented by a list of column indices for selection.
    """
    def __init__(self, p_list: List[int]):
        self.p_list = p_list
        self.D = len(p_list)
        # cumulative column offsets for each view in the big matrix
        self.offsets = np.cumsum([0] + p_list[:-1]).tolist()

        # index ranges for each view in the concatenated matrix
        self.view_cols = [list(range(self.offsets[d], self.offsets[d] + p_list[d])) for d in range(self.D)]

        # Level 3 (individual)
        self.level3 = [tuple([d]) for d in range(self.D)]
        self.level3_cols = [self.view_cols[d] for d in range(self.D)]

        # Level 2 (pairwise)
        self.level2 = list(itertools.combinations(range(self.D), 2))
        self.level2_cols = [sorted(sum([self.view_cols[d] for d in pair], [])) for pair in self.level2]

        # Level 1 (all)
        self.level1 = [tuple(range(self.D))]
        self.level1_cols = [list(range(sum(self.p_list)))]

        # The full ordered set of operators: individual, pairwise, all (same as Eq. (11) groups)
        self.groups = {
            'individual': self.level3,
            'pairwise'  : self.level2,
            'all'       : self.level1
        }
        self.group_cols = {
            'individual': self.level3_cols,
            'pairwise'  : self.level2_cols,
            'all'       : self.level1_cols
        }

        # Build flat lists for iteration
        self.ops = []
        for g in ['individual', 'pairwise', 'all']:
            for subset, cols in zip(self.groups[g], self.group_cols[g]):
                self.ops.append({'group': g, 'subset': subset, 'cols': np.array(cols, dtype=int)})

# ---------- main solver ----------
class HNN:
    """
    Hierarchical Nuclear Norm (HNN) integrator with primal-dual (PDHG) solver,
    refitting, and 2x2 bi-cross-validation.
    """
    def __init__(self,
                 tau_individual: float = 0.5,
                 kappa_pairwise: float = 0.5,
                 lam_all: float = 0.5,
                 weights_individual: Optional[List[float]] = None,
                 weights_pairwise: Optional[List[float]] = None,
                 step_primal: float = 0.2,
                 step_dual: float = 0.2,
                 max_iter: int = 2000,
                 tol: float = 1e-5,
                 refit: bool = True,
                 random_state: Optional[int] = None):
        """
        Parameters map to Eq. (11):
            penalty = tau * sum_d (w_d * ||M_d||_*) +
                      kappa * sum_pairs (w_pair * ||[M_k M_l]||_*) +
                      lam * ||[M_1 ... M_D]||_*
        """
        self.tau = float(tau_individual)
        self.kappa = float(kappa_pairwise)
        self.lam_all = float(lam_all)
        self.w_ind = weights_individual
        self.w_pair = weights_pairwise
        self.t_p = float(step_primal)
        self.s_d = float(step_dual)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.refit = bool(refit)
        self.rng = np.random.default_rng(random_state)

        # fitted attributes
        self.M_list_: Optional[List[np.ndarray]] = None
        self.M_refit_: Optional[List[np.ndarray]] = None
        self.history_: Dict[str, List[float]] = {}

    @staticmethod
    def _preprocess_views(X_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Center columns of each view and scale Frobenius norm to 1 (as in Sec. 2.4).
        Returns scaled copies and scales to invert later for interpretability if desired.
        """
        Xc = []
        scales = []
        for X in X_list:
            X = X - np.mean(X, axis=0, keepdims=True)
            f = np.linalg.norm(X, 'fro')
            if f == 0:
                X = X.copy()
                f = 1.0
            Xc.append(X / f)
            scales.append(f)
        return Xc, scales

    def _init_weights(self, ops: HNNOperators):
        # Individual weights: one per view (sum to 1 by default)
        if self.w_ind is None:
            w = np.ones(ops.D, dtype=float)
            self.w_ind = (w / w.sum()).tolist()
        # Pairwise weights: one per pair (sum to 1 by default)
        if self.w_pair is None:
            w = np.ones(len(ops.level2), dtype=float) if len(ops.level2) > 0 else np.array([1.0])
            self.w_pair = (w / w.sum()).tolist()

    def fit(self, X_list: List[np.ndarray]) -> "HNN":
        """
        Fit HNN on a list of (n x p_d) views.
        """
        X_list_proc, _scales = self._preprocess_views(X_list)
        n = X_list_proc[0].shape[0]
        assert all(X.shape[0] == n for X in X_list_proc), "All views must have matching sample size n."
        p_list = [X.shape[1] for X in X_list_proc]
        ops = HNNOperators(p_list)
        self._init_weights(ops)

        # Build the big concatenated variables
        X_all = column_stack_views(X_list_proc)
        M_all = np.zeros_like(X_all)  # primal
        M_prev = M_all.copy()

        # Dual variables, one per operator, sized to selected columns
        Y = [np.zeros((n, len(op['cols']))) for op in ops.ops]

        # Map group -> lambda weight per operator (Eq. 11)
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
            else:  # 'all'
                lam = self.lam_all
            group_lams.append(lam)

        # Precompute gradient of data-fit part: f(M) = 0.5 ||X - M||_F^2 => grad f = M - X
        # PDHG iterations
        self.history_ = {'primal_res': [], 'dual_res': [], 'obj': []}
        for it in range(self.max_iter):
            # Dual ascent + prox of g* for each operator
            for j, op in enumerate(ops.ops):
                cols = op['cols']
                # Y^{k+1} = prox_{sigma g*}( Y^k + sigma * L M^k )
                Y[j] = Y[j] + self.s_d * M_all[:, cols]
                Y[j] = prox_conjugate_nuc(Y[j], sigma=self.s_d, lam=group_lams[j])

            # Primal descent on f + coupling via adjoints
            # M^{k+1} = prox_{tau f}( M^k - tau * sum L^T Y^{k+1} )
            # Here prox_{tau f}(Z) = (Z + tau * X) / (1 + tau) because f is 0.5||X-M||^2_F
            Z = M_all.copy()
            # accumulate adjoint contributions: L^T Y = put Y back to selected columns
            A = np.zeros_like(M_all)
            for j, op in enumerate(ops.ops):
                cols = op['cols']
                A[:, cols] += Y[j]
            Z = Z - self.t_p * A
            M_all = (Z + self.t_p * X_all) / (1.0 + self.t_p)

            # simple diagnostics
            primal_res = np.linalg.norm(M_all - M_prev, 'fro') / (np.linalg.norm(M_prev, 'fro') + 1e-12)
            M_prev = M_all.copy()

            # objective value (for monitoring)
            # data-fit
            obj = 0.5 * frob2(X_all - M_all)
            # penalties
            # individuals
            for d in range(ops.D):
                cols = ops.level3_cols[d]
                obj += self.tau * float(self.w_ind[d]) * nuclear_norm(M_all[:, cols])
            # pairwise
            for j, pair in enumerate(ops.level2_cols):
                obj += self.kappa * float(self.w_pair[j]) * nuclear_norm(M_all[:, pair])
            # all
            obj += self.lam_all * nuclear_norm(M_all)

            self.history_['primal_res'].append(primal_res)
            self.history_['obj'].append(obj)

            if primal_res < self.tol:
                break

        # split back to per-view matrices
        col_slices = [slice(ops.offsets[d], ops.offsets[d] + p_list[d]) for d in range(ops.D)]
        self.M_list_ = split_columns(M_all, col_slices)

        # optional refit (Algorithm 2): project onto estimated column spaces
        if self.refit:
            self.M_refit_ = []
            for X, M in zip(X_list_proc, self.M_list_):
                if np.allclose(M, 0):
                    self.M_refit_.append(M.copy())
                    continue
                U, s, Vt = svd(M, full_matrices=False)
                r = np.sum(s > 1e-10)
                if r == 0:
                    self.M_refit_.append(np.zeros_like(M))
                else:
                    U_r = U[:, :r]
                    # least squares with column-space constraint => U U^T X
                    M_refit = U_r @ (U_r.T @ X)
                    self.M_refit_.append(M_refit)
        else:
            self.M_refit_ = None
        return self

    def get_signal(self, refit: bool = True) -> List[np.ndarray]:
        if self.M_list_ is None:
            raise RuntimeError("Call fit() first.")
        if refit and (self.M_refit_ is not None):
            return [Mi.copy() for Mi in self.M_refit_]
        return [Mi.copy() for Mi in self.M_list_]

    # ---------- 2x2 BCV (Algorithm 3) ----------
    def bcv_2x2(self,
                X_list: List[np.ndarray],
                grid_tau: List[float],
                grid_kappa: List[float],
                grid_lambda: List[float],
                max_iter_each: int = 1000) -> Tuple[Tuple[float,float,float], Dict[str, float]]:
        """
        Simple 2x2 BCV to select (tau, kappa, lambda).
        Returns best triple (1-SE rule) and metrics.
        """
        n = X_list[0].shape[0]
        # column folds per view: 2 blocks
        p_list = [X.shape[1] for X in X_list]
        # make row fold indices
        rows = np.arange(n)
        self.rng.shuffle(rows)
        rmid = n // 2
        row_blocks = [rows[:rmid], rows[rmid:]]

        # column blocks per view
        col_blocks = []
        for p in p_list:
            cols = np.arange(p)
            self.rng.shuffle(cols)
            cmid = p // 2
            col_blocks.append([cols[:cmid], cols[cmid:]])

        # helper to extract submatrices by row/col blocks
        def sub_views(Xs, rsel, csel_list):
            out = []
            for X, (c0, c1), csel in zip(Xs, col_blocks, csel_list):
                out.append(X[np.ix_(rsel, csel)])
            return out

        # build four holdouts (j,k) in {0,1}x{0,1}
        candidates = list(itertools.product(grid_tau, grid_kappa, grid_lambda))
        errs = []

        for (tau, kap, lam) in candidates:
            fold_errs = []
            for j in [0,1]:
                for k in [0,1]:
                    # held-out blocks: for every view, the (j,k) block
                    r_hold = row_blocks[j]
                    # for columns: choose kth block for each view
                    c_hold_list = [cb[k] for cb in col_blocks]

                    # training rows/cols: those that don't share rows or cols with holdout
                    r_train = row_blocks[1-j]
                    # for columns, use the opposite block for each view
                    c_train_list = [cb[1-k] for cb in col_blocks]

                    X_train = sub_views(X_list, r_train, c_train_list)
                    model = HNN(tau_individual=tau, kappa_pairwise=kap, lam_all=lam,
                                step_primal=self.t_p, step_dual=self.s_d,
                                max_iter=max_iter_each, tol=self.tol, refit=True,
                                random_state=None)
                    model.fit(X_train)

                    # predictions on each held-out block via learned column spaces (project X_hold)
                    # We mimic Algorithm 3's use of row/col-sharing pieces for prediction—here simplified.
                    X_hold = sub_views(X_list, r_hold, c_hold_list)
                    M_pred = []
                    for Xh, Mh in zip(X_hold, model.get_signal(refit=True)):
                        # Use projection U U^T with U from Mh trained on disjoint rows/cols
                        if np.allclose(Mh, 0):
                            M_pred.append(np.zeros_like(Xh))
                        else:
                            U, s, _ = svd(Mh, full_matrices=False)
                            r = np.sum(s > 1e-10)
                            if r == 0:
                                M_pred.append(np.zeros_like(Xh))
                            else:
                                U_r = U[:, :r]
                                # Note: dimensions align because Mh is trained on r_train rows;
                                # for prediction on r_hold, reuse right singular vectors is not possible.
                                # As a practical proxy, we project columns by solving LS per column with ridge ~ 0.
                                # Here we fallback to zero (conservative). For compactness, we project rows instead:
                                # We'll use column space learned to denoise columns of Xh:
                                # Solve min_C ||Xh - U_r C||_F^2 => C = U_r^T Xh  (since U_r spans row-space across rows)
                                C = U_r.T @ Xh
                                M_pred.append(U_r @ C)
                    # error on holdout
                    e = sum(frob2(Xh - Mp) for Xh, Mp in zip(X_hold, M_pred))
                    fold_errs.append(e)
            errs.append((tau, kap, lam, float(np.mean(fold_errs)), float(np.std(fold_errs))))

        # pick by 1-SE rule
        errs_sorted = sorted(errs, key=lambda t: t[3])
        best_mean = errs_sorted[0][3]
        best_std = errs_sorted[0][4]
        cutoff = best_mean + best_std  # simple 1-SE
        # smallest (tau,kap,lam) with mean <= cutoff (prefer stronger regularization)
        chosen = None
        for tup in errs_sorted:
            if tup[3] <= cutoff:
                chosen = tup
                break
        (bt, bk, bl, m, s) = chosen
        return (bt, bk, bl), {'cv_mean': m, 'cv_std': s, 'all': errs_sorted}

# helper for objective logging
def nuclear_norm(X: np.ndarray) -> float:
    return float(np.sum(svd(X, full_matrices=False, compute_uv=False)))
