"""Replicates the logic of poisson_glm_from_scratch.ipynb (train_poisson_glm) and compares with statsmodels."""
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRATCH = Path(tempfile.gettempdir()) / "mtpl_audit_cache"  # cache outside the repo
SCRATCH.mkdir(exist_ok=True)
df_f, _ = pickle.loads((SCRATCH / "raw.pkl").read_bytes())

# Notebook preprocessing: four columns, numeric, exposure clipped at 1, dropna. No other filter.
d = df_f[["ClaimNb", "Exposure", "DrivAge", "VehPower"]].copy()
for c in d.columns:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["Exposure"] = d["Exposure"].clip(upper=1.0)
d = d.dropna()
y = d["ClaimNb"].values.astype(float)
exposure = d["Exposure"].values.astype(float)
X = np.hstack([np.ones((len(d), 1)), d[["DrivAge", "VehPower"]].values.astype(float)])

np.random.seed(42)
n_total = X.shape[0]
n_train = int(0.8 * n_total)
idx = np.random.permutation(n_total)
tr = idx[:n_train]
Xtr, ytr, etr = X[tr], y[tr], exposure[tr]


def irls(X, y, exposure, n_iter):
    beta = np.zeros(X.shape[1])
    hist = []
    for _ in range(n_iter):
        eta = X @ beta
        mu = exposure * np.exp(eta)
        z = eta + (y - mu) / mu
        XTWX = X.T @ (mu[:, None] * X)
        XTWz = X.T @ (mu * z)
        beta = np.linalg.solve(XTWX, XTWz)
        hist.append(beta.copy())
    return beta, hist


res = sm.GLM(ytr, Xtr, family=sm.families.Poisson(), offset=np.log(etr)).fit(tol=1e-12, maxiter=100)
print("rows:", len(d), " train:", n_train)
print("statsmodels beta:", res.params, " iterations:", res.fit_history["iteration"])
beta10, hist = irls(Xtr, ytr, etr, 25)
for k in [1, 2, 3, 5, 8, 10, 12, 15, 20, 25]:
    b = hist[k - 1]
    print(f"iter {k:2d}: beta = {np.array2string(b, precision=8)}  max abs diff vs statsmodels = {np.max(np.abs(b - res.params)):.3e}"
          f"  max rel diff = {np.max(np.abs(b - res.params) / np.abs(res.params)):.3e}")
