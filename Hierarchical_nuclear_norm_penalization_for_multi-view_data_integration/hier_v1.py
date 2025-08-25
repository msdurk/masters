# demo.py
import numpy as np
from impl import HNN

# ----------------------------
# Step 1. Create synthetic views
# ----------------------------
n = 50          # samples
p1, p2, p3 = 20, 25, 15   # features per view
rank_shared = 3           # shared latent rank

rng = np.random.default_rng(123)

# Shared latent structure (n x r) times (r x p_d) loading matrices
U = rng.standard_normal((n, rank_shared))

V1 = rng.standard_normal((rank_shared, p1))
V2 = rng.standard_normal((rank_shared, p2))
V3 = rng.standard_normal((rank_shared, p3))

# Generate signal + Gaussian noise
X1 = U @ V1 + 0.2 * rng.standard_normal((n, p1))
X2 = U @ V2 + 0.2 * rng.standard_normal((n, p2))
X3 = U @ V3 + 0.2 * rng.standard_normal((n, p3))

views = [X1, X2, X3]

# ----------------------------
# Step 2. Fit HNN
# ----------------------------
model = HNN(
    tau_individual=0.6,
    kappa_pairwise=0.4,
    lam_all=0.3,
    step_primal=0.2,
    step_dual=0.2,
    max_iter=1000,
    tol=1e-5,
    refit=True,
    random_state=42,
)

model.fit(views)

# ----------------------------
# Step 3. Inspect results
# ----------------------------
M1, M2, M3 = model.get_signal(refit=True)

print("Original X1 shape:", X1.shape, "Recovered M1 shape:", M1.shape)
print("Frobenius norm of X1:", np.linalg.norm(X1, 'fro'))
print("Frobenius norm of M1:", np.linalg.norm(M1, 'fro'))

print("\nObjective history (last 5 values):")
print(model.history_['obj'][-5:])

# ----------------------------
# Step 4. (Optional) Run 2x2 BCV to pick penalties
# ----------------------------
grid_tau   = [0.3, 0.6, 0.9]
grid_kappa = [0.3, 0.6, 0.9]
grid_lambda= [0.2, 0.5, 0.8]

best, info = model.bcv_2x2(views, grid_tau, grid_kappa, grid_lambda, max_iter_each=500)
print("\nSelected (tau, kappa, lambda) via BCV:", best, "CV mean error:", info['cv_mean'])
