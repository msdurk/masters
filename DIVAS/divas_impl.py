"""
DIVAS: Data Integration Via Analysis of Subspaces (Python)

A practical, self-contained Python implementation of the DIVAS methodology
(Prothero et al., 2024) for discovering partially shared joint structure across
multiple data blocks. Rows index *traits* (features) and columns index *objects*
(samples), matching the paper's convention.

Key features
------------
- Per-block signal subspace estimation via SVD with noise/ rank heuristics
- Rotational bootstrap to compute trait/object angle bounds (φ̂, ψ̂) and a
  filtered signal rank ř
- Greedy/DC-style search for joint score vectors across arbitrary block
  collections, constrained by φ̂/ψ̂ and exclusivity vs. excluded blocks
- Signal reconstruction: per-collection shared scores (V) and per-block loadings
  (L), with reconstructed signal A = L Vᵀ for participating blocks
- Light diagnostics (ENC, ECT, bounds)

Dependencies
------------
- numpy (required)
- scipy (optional) — for slightly more stable QR/SVD; falls back to NumPy

Usage
---------------
>>> import numpy as np
>>> from divas_impl import DIVAS, Centering
>>> rng = np.random.RandomState(0)
>>> n = 100                                # objects
>>> d1, d2 = 50, 80                        # traits per block
>>> V_shared = orth(rng.normal(size=(n, 2)))
>>> L1, L2 = rng.normal(size=(d1,2)), rng.normal(size=(d2,2))
>>> A1, A2 = L1 @ V_shared.T, L2 @ V_shared.T
>>> X1, X2 = A1 + 0.5*rng.normal(size=A1.shape), A2 + 0.5*rng.normal(size=A2.shape)
>>> model = DIVAS(M_boot=60, random_state=0, centering=Centering(object_center=True))
>>> result = model.fit([X1, X2])
>>> [(c.collection, c.V_scores.shape) for c in result.joint_components]
[((0, 1), (100, 2))]
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from scipy.linalg import qr as scipy_qr
except Exception:  # pragma: no cover
    scipy_qr = None

# ---------------- Linear algebra helpers ----------------

def orth(A: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis for the columns of A using QR or SVD.
    Robust to rank deficiency. If A has zero size, returns A.
    """
    if A.size == 0:
        return A
    if scipy_qr is not None:
        Q, R = scipy_qr(A, mode="economic")
        if R.size:
            tol = np.finfo(A.dtype if A.dtype.kind in "fc" else float).eps * max(A.shape) * np.max(np.abs(np.diag(R)))
            keep = np.where(np.abs(np.diag(R)) > tol)[0]
            return Q[:, keep]
        return Q
    # Fallback: SVD
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if s.size == 0:
        return np.zeros((A.shape[0], 0), dtype=A.dtype)
    tol = np.finfo(A.dtype if A.dtype.kind in "fc" else float).eps * max(A.shape) * s.max()
    r = int((s > tol).sum())
    return U[:, :r] if r > 0 else np.zeros((A.shape[0], 0), dtype=A.dtype)


def principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Principal angles (deg) between span(U) and span(V). U, V are (m, r1/ r2).
    Pads with 90° if ranks differ. Returns sorted ascending angles.
    """
    if U.size == 0 or V.size == 0:
        dim = max(U.shape[1] if U.ndim == 2 else 0, V.shape[1] if V.ndim == 2 else 0)
        return np.full(dim, 90.0, dtype=float)
    Uo, Vo = orth(U), orth(V)
    s = np.linalg.svd(Uo.T @ Vo, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    ang = np.degrees(np.arccos(s))
    if Uo.shape[1] != Vo.shape[1]:
        ang = np.concatenate([ang, np.full(abs(Uo.shape[1]-Vo.shape[1]), 90.0)])
    return np.sort(ang)


def angle_to_subspace(v: np.ndarray, V: np.ndarray) -> float:
    """Smallest angle (deg) between unit vector v and span(V)."""
    if V.size == 0:
        return 90.0
    v = v.reshape(-1)
    v = v / (np.linalg.norm(v) + 1e-12)
    c = float(np.linalg.norm(V.T @ v))
    c = float(np.clip(c, 0.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def angle_vector_to_subspace(x: np.ndarray, U: np.ndarray) -> float:
    """Angle (deg) between nonzero vector x and span(U)."""
    if U.size == 0 or np.linalg.norm(x) == 0:
        return 90.0
    x = x / (np.linalg.norm(x) + 1e-12)
    c = float(np.linalg.norm(U.T @ x))
    c = float(np.clip(c, 0.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def rand_orthonormal(m: int, r: int, rng: np.random.RandomState, orth_to: Optional[np.ndarray] = None) -> np.ndarray:
    """Random orthonormal (m, r). If orth_to is given (m, p), ensure columns are
    orthogonal to span(orth_to).
    """
    if r <= 0:
        return np.zeros((m, 0))
    G = rng.normal(size=(m, r + (orth_to.shape[1] if orth_to is not None and orth_to.size else 0)))
    if orth_to is not None and orth_to.size:
        Q = orth(orth_to)
        G = G - Q @ (Q.T @ G)
    Q2 = orth(G)
    if Q2.shape[1] < r:  # try to fill missing dims
        G2 = rng.normal(size=(m, r))
        if orth_to is not None and orth_to.size:
            Q = orth(orth_to)
            G2 = G2 - Q @ (Q.T @ G2)
        Q2 = orth(np.hstack([Q2, G2]))
    return Q2[:, :r]


def top_eigvec(M: np.ndarray, max_iter: int = 400, tol: float = 1e-8, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Power iteration for the top eigenvector of a symmetric matrix M."""
    r = rng or np.random.RandomState(0)
    v = r.normal(size=(M.shape[0],))
    v /= np.linalg.norm(v) + 1e-12
    last_val = None
    for _ in range(max_iter):
        w = M @ v
        nrm = np.linalg.norm(w)
        if nrm == 0:
            v = r.normal(size=v.shape)
            v /= np.linalg.norm(v) + 1e-12
            continue
        v = w / nrm
        val = float(v @ (M @ v))
        if last_val is not None and abs(val - last_val) < tol * (abs(last_val) + 1e-12):
            break
        last_val = val
    return v

# ---------------- Centering ----------------

@dataclass
class Centering:
    object_center: bool = True  # subtract row means (across columns)
    trait_center: bool = False  # subtract column means (across rows)


def apply_centering(X: np.ndarray, centering: Centering) -> tuple[np.ndarray, dict]:
    Xc = X.copy()
    info = {}
    if centering.object_center:
        mu_rows = Xc.mean(axis=1, keepdims=True)
        Xc -= mu_rows
        info["row_means"] = mu_rows
    if centering.trait_center:
        mu_cols = Xc.mean(axis=0, keepdims=True)
        Xc -= mu_cols
        info["col_means"] = mu_cols
    return Xc, info

# ---------------- Signal extraction ----------------

@dataclass
class SignalEstimate:
    U: np.ndarray  # (d, r̂)
    V: np.ndarray  # (n, r̂)
    S: np.ndarray  # (r̂,)
    rank_hat: int
    X_centered: np.ndarray
    sigma_hat: float


def _estimate_sigma_via_median(d: int, n: int, svals: np.ndarray, sims: int = 20, rng: Optional[np.random.RandomState] = None) -> float:
    """Estimate noise sigma by calibrating the median singular value vs. simulated
    pure-noise matrices of size (d, n). Fast and fairly robust.
    """
    r = rng or np.random.RandomState(0)
    med_data = float(np.median(svals))
    meds = []
    for _ in range(sims):
        N = r.normal(size=(d, n))
        sv = np.linalg.svd(N, compute_uv=False)
        meds.append(np.median(sv))
    med_noise = float(np.median(meds))
    return med_data / (med_noise + 1e-12)


def _rank_from_mp_edge(d: int, n: int, svals: np.ndarray, sigma_hat: float, fudge: float = 1.02) -> int:
    """Heuristic rank via Marchenko–Pastur edge: keep singular values above
    sigma_hat * (sqrt(n) + sqrt(d)) * fudge.
    """
    edge = sigma_hat * (math.sqrt(n) + math.sqrt(d)) * fudge
    return int(np.sum(svals > edge))


def signal_extract(X: np.ndarray, centering: Centering, rng: np.random.RandomState, fudge: float = 1.02) -> SignalEstimate:
    Xc, _ = apply_centering(X, centering)
    d, n = Xc.shape
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    sigma_hat = _estimate_sigma_via_median(d, n, s, rng=rng)
    r_hat = _rank_from_mp_edge(d, n, s, sigma_hat, fudge=fudge)
    if r_hat == 0:
        U_keep = np.zeros((d, 0))
        V_keep = np.zeros((n, 0))
        s_keep = np.zeros((0,))
    else:
        U_keep = U[:, :r_hat]
        V_keep = Vt.T[:, :r_hat]
        s_keep = s[:r_hat].copy()
    return SignalEstimate(U=U_keep, V=V_keep, S=s_keep, rank_hat=r_hat, X_centered=Xc, sigma_hat=sigma_hat)

# ---------------- Rotational bootstrap & bounds ----------------

@dataclass
class BootstrapBounds:
    phi_trait_deg: float  # ϕ̂
    psi_object_deg: float # ψ̂
    r_filtered: int       # ř
    V_filtered: np.ndarray
    U_filtered: np.ndarray
    theta0_deg: float


def _random_direction_angle_bound(V: np.ndarray, n: int, r: int, rng: np.random.RandomState, quantile: float = 0.05, samples: int = 2000) -> float:
    if r == 0:
        return 90.0
    if V.size == 0:
        V = rand_orthonormal(n, r, rng)
    angles = []
    for _ in range(samples):
        v = rng.normal(size=(n,))
        v /= np.linalg.norm(v) + 1e-12
        angles.append(angle_to_subspace(v, V))
    return float(np.quantile(angles, quantile))


def rotational_bootstrap(signal: SignalEstimate, centering: Centering, xi: float, alpha: float, M: int, rng: np.random.RandomState) -> BootstrapBounds:
    d = signal.U.shape[0]
    n = signal.V.shape[0]
    r_hat = signal.rank_hat
    if r_hat == 0:
        zV = np.zeros((n, 0))
        zU = np.zeros((d, 0))
        return BootstrapBounds(90.0, 90.0, 0, zV, zU, 90.0)

    # Simulate residual noise with same centering
    E = rng.normal(scale=signal.sigma_hat, size=(d, n))
    E, _ = apply_centering(E, centering)

    theta0 = _random_direction_angle_bound(signal.V, n, r_hat, rng, quantile=0.05, samples=2000)

    traitAngles = np.zeros((M, r_hat))
    objAngles   = np.zeros((M, r_hat))

    const_trait = np.ones((n, 1)) / math.sqrt(n)
    const_obj   = np.ones((d, 1)) / math.sqrt(d)

    for m in range(M):
        U0 = rand_orthonormal(d, r_hat, rng, orth_to=(const_obj if centering.object_center else None))
        V0 = rand_orthonormal(n, r_hat, rng, orth_to=(const_trait if centering.trait_center else None))
        A0 = U0 @ np.diag(signal.S if signal.S.size == r_hat else np.ones(r_hat)) @ V0.T
        X0 = A0 + E
        U1, _, V1t = np.linalg.svd(X0, full_matrices=False)
        U1 = U1[:, :r_hat]
        V1 = V1t.T[:, :r_hat]
        for j in range(1, r_hat+1):
            ang_t = principal_angles(V0, V1[:, :j])
            ang_o = principal_angles(U0, U1[:, :j])
            traitAngles[m, j-1] = float(ang_t.max() if ang_t.size else 90.0)
            objAngles[m, j-1]   = float(ang_o.max() if ang_o.size else 90.0)

    trait_sorted = np.sort(traitAngles, axis=0)
    obj_sorted   = np.sort(objAngles, axis=0)
    idx = min(max(int(math.floor(alpha * M)) - 1, 0), M-1)

    r_filtered = 0
    for j in range(1, r_hat+1):
        if trait_sorted[idx, j-1] < xi * theta0 and obj_sorted[idx, j-1] < xi * theta0:
            r_filtered = j
    if r_filtered == 0:
        r_filtered = min(1, r_hat)

    phi_hat = float(trait_sorted[idx, r_filtered-1])
    psi_hat = float(obj_sorted[idx, r_filtered-1])

    Ufil = signal.U[:, :r_filtered]
    Vfil = signal.V[:, :r_filtered]
    return BootstrapBounds(phi_trait_deg=phi_hat, psi_object_deg=psi_hat, r_filtered=r_filtered,
                           V_filtered=Vfil, U_filtered=Ufil, theta0_deg=theta0)

# ---------------- Main DIVAS ----------------

@dataclass
class BlockResult:
    U: np.ndarray
    V: np.ndarray
    S: np.ndarray
    rank_hat: int
    r_filtered: int
    phi_deg: float
    psi_deg: float
    theta0_deg: float
    V_filtered: np.ndarray
    U_filtered: np.ndarray
    X_centered: np.ndarray
    sigma_hat: float
    projector_trait: np.ndarray  # V_filtered V_filteredᵀ

@dataclass
class JointComponent:
    collection: Tuple[int, ...]              # indices of participating blocks
    V_scores: np.ndarray                     # (n, r_i)
    L_loadings: Dict[int, np.ndarray]        # k -> (d_k, r_i)
    Ai_per_block: Dict[int, np.ndarray]      # k -> (d_k, n)

@dataclass
class DivasResult:
    joint_components: List[JointComponent]
    block_results: Dict[int, BlockResult]
    centering: Centering


class DIVAS:
    def __init__(self,
                 alpha: float = 0.95,
                 xi: float = 0.382,          # golden-ratio-inspired default from paper
                 M_boot: int = 80,           # bootstrap reps (increase for more precision)
                 centering: Centering = Centering(object_center=True, trait_center=False),
                 exclude_weight: float = 0.5,# penalty λ for excluded blocks
                 random_state: Optional[int] = 0):
        self.alpha = alpha
        self.xi = xi
        self.M_boot = M_boot
        self.centering = centering
        self.exclude_weight = float(exclude_weight)
        self.rng = np.random.RandomState(random_state)
        self._fitted = False
        self._result: Optional[DivasResult] = None

    def fit(self, blocks: List[np.ndarray]) -> DivasResult:
        """Run DIVAS on a list of blocks [X1, X2, ...], each shaped (d_k, n).
        All blocks must share the same number of columns (objects n).
        Returns a DivasResult containing joint components and per-block summaries.
        """
        if not blocks:
            raise ValueError("No blocks provided.")
        n = blocks[0].shape[1]
        for b in blocks:
            if b.shape[1] != n:
                raise ValueError("All blocks must have the same number of columns (objects).")
        K = len(blocks)

        # 1) Per-block signal estimate and rotational bootstrap
        block_results: Dict[int, BlockResult] = {}
        for k, Xk in enumerate(blocks):
            sig = signal_extract(Xk, self.centering, self.rng)
            boot = rotational_bootstrap(sig, self.centering, self.xi, self.alpha, self.M_boot, self.rng)
            Pk = boot.V_filtered @ boot.V_filtered.T if boot.V_filtered.size else np.zeros((n, n))
            block_results[k] = BlockResult(U=sig.U, V=sig.V, S=sig.S, rank_hat=sig.rank_hat,
                                           r_filtered=boot.r_filtered, phi_deg=boot.phi_trait_deg, psi_deg=boot.psi_object_deg,
                                           theta0_deg=boot.theta0_deg, V_filtered=boot.V_filtered, U_filtered=boot.U_filtered,
                                           X_centered=sig.X_centered, sigma_hat=sig.sigma_hat,
                                           projector_trait=Pk)

        # 2) Greedy/DC-style search over collections
        found_V: Dict[Tuple[int, ...], np.ndarray] = {}

        def supersets_of(coll: Tuple[int, ...]) -> List[Tuple[int, ...]]:
            sc = set(coll)
            return [key for key in found_V.keys() if sc.issubset(set(key)) and sc != set(key)]

        for size in range(K, 0, -1):
            for coll in itertools.combinations(range(K), size):
                # Build nullspace projector against supersets already found
                Vsup = [found_V[s] for s in supersets_of(coll) if found_V[s].size]
                Vsup_cat = np.hstack(Vsup) if Vsup else np.zeros((n, 0))
                Qsup = orth(Vsup_cat) if Vsup else np.zeros((n, 0))
                Pnull = np.eye(n) - (Qsup @ Qsup.T if Qsup.size else 0)

                # Objective M = sum_in Pk - λ * sum_out Pk
                Pin = sum((block_results[k].projector_trait for k in coll), start=np.zeros((n, n)))
                Pout = sum((block_results[k].projector_trait for k in range(K) if k not in coll), start=np.zeros((n, n)))
                M = Pin - self.exclude_weight * Pout
                Mtil = Pnull @ M @ Pnull

                Vi_list: List[np.ndarray] = []
                max_new = max(block_results[k].r_filtered for k in coll)
                attempts = 0
                while len(Vi_list) < max_new and attempts < 8:
                    if Vi_list:
                        Vcat = orth(np.hstack([Qsup] + Vi_list)) if Qsup.size else orth(np.hstack(Vi_list))
                        Pn2 = np.eye(n) - (Vcat @ Vcat.T if Vcat.size else 0)
                        Mcur = Pn2 @ Mtil @ Pn2
                    else:
                        Mcur = Mtil
                    v = top_eigvec(Mcur, rng=self.rng)
                    v /= np.linalg.norm(v) + 1e-12

                    ok = True
                    # Angle constraints for included blocks
                    for k in coll:
                        Vk = block_results[k].V_filtered
                        Uk = block_results[k].U_filtered
                        phi = block_results[k].phi_deg
                        psi = block_results[k].psi_deg
                        if angle_to_subspace(v, Vk) > phi:
                            ok = False; break
                        xkv = block_results[k].X_centered @ v
                        if angle_vector_to_subspace(xkv, Uk) > psi:
                            ok = False; break
                    # Exclusivity vs. excluded blocks
                    if ok:
                        for k in range(K):
                            if k in coll:
                                continue
                            Vk = block_results[k].V_filtered
                            phi_ex = block_results[k].phi_deg
                            if angle_to_subspace(v, Vk) <= phi_ex:
                                ok = False; break
                    if ok:
                        Vi_list.append(v.reshape(-1, 1))
                        attempts = 0
                    else:
                        self.exclude_weight *= 1.15
                        attempts += 1

                Vi = orth(np.hstack(Vi_list)) if Vi_list else np.zeros((n, 0))
                found_V[coll] = Vi

        # 3) Reconstruction per component
        joint_components: List[JointComponent] = []
        for coll, Vi in found_V.items():
            if Vi.size == 0:
                continue
            Ls: Dict[int, np.ndarray] = {}
            Ais: Dict[int, np.ndarray] = {}
            G = np.linalg.pinv(Vi.T @ Vi)
            for k in coll:
                Xk = block_results[k].X_centered
                Lk = Xk @ Vi @ G
                Ak = Lk @ Vi.T
                Ls[k] = Lk
                Ais[k] = Ak
            joint_components.append(JointComponent(collection=coll, V_scores=Vi, L_loadings=Ls, Ai_per_block=Ais))

        res = DivasResult(joint_components=joint_components, block_results=block_results, centering=self.centering)
        self._fitted = True
        self._result = res
        return res

    # ---------------- Diagnostics ----------------

    @staticmethod
    def enc(v: np.ndarray) -> float:
        """Effective Number of Cases for a unit-norm scores vector v."""
        v = v / (np.linalg.norm(v) + 1e-12)
        return float(1.0 / (np.sum(v**4) + 1e-12))

    @staticmethod
    def ect(l: np.ndarray) -> float:
        """Effective Contribution of Traits (percentage) for a loadings vector l."""
        d = l.shape[0]
        num = (np.sum(l**2))**2
        den = d * (np.sum(l**4) + 1e-12)
        return float(100.0 * (num / (den + 1e-12)))

    def diagnostics(self) -> Dict:
        if not self._fitted or self._result is None:
            raise RuntimeError("Call fit() first.")
        info = {"blocks": {}, "components": []}
        for k, br in self._result.block_results.items():
            info["blocks"][k] = {
                "rank_hat": br.rank_hat,
                "r_filtered": br.r_filtered,
                "phi_deg": br.phi_deg,
                "psi_deg": br.psi_deg,
                "theta0_deg": br.theta0_deg,
            }
        for comp in self._result.joint_components:
            Vi = comp.V_scores
            encs = [self.enc(Vi[:, i]) for i in range(Vi.shape[1])] if Vi.size else []
            ects = {k: [self.ect(Lk[:, i]) for i in range(Lk.shape[1])] for k, Lk in comp.L_loadings.items()}
            info["components"].append({"collection": comp.collection, "scores_dim": Vi.shape[1], "ENC": encs, "ECT": ects})
        return info


# Convenience re-exports
__all__ = [
    "DIVAS", "Centering", "DivasResult", "JointComponent", "BlockResult",
    "orth", "principal_angles", "angle_to_subspace", "angle_vector_to_subspace"
]
