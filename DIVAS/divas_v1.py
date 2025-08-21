import numpy as np
from divas_impl import DIVAS, Centering

# Fake example with 2 blocks sharing a 2D trait subspace
rng = np.random.RandomState(0)
n = 100
d1, d2 = 50, 80
V_shared = np.linalg.qr(rng.normal(size=(n, 2)))[0]
L1, L2 = rng.normal(size=(d1, 2)), rng.normal(size=(d2, 2))
A1, A2 = L1 @ V_shared.T, L2 @ V_shared.T
X1, X2 = A1 + 0.5*rng.normal(size=A1.shape), A2 + 0.5*rng.normal(size=A2.shape)

model = DIVAS(M_boot=80, random_state=0, centering=Centering(object_center=True))
result = model.fit([X1, X2])

# Joint components discovered (collections and score dims)
[(c.collection, c.V_scores.shape) for c in result.joint_components]

# Optional diagnostics
print(model.diagnostics())
