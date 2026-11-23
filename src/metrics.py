"""Metrics shared by every model. Frequency metrics are primary.

Conventions: `pred_rate` is a prediction per unit of exposure (claims or euros per policy-year),
`actual` is the observed total for the policy (claim count or loss), `exposure` is in policy-years.
"""

import numpy as np
import pandas as pd


def poisson_deviance(y, mu):
    """Total Poisson deviance, 2 * sum(y log(y / mu) - (y - mu)), with y log y = 0 at y = 0."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    ylogy = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(2.0 * np.sum(ylogy - (y - mu)))


def mean_poisson_deviance(claims, pred_rate, exposure):
    """Exposure-weighted mean Poisson deviance: total deviance of the counts per unit of exposure."""
    exposure = np.asarray(exposure, dtype=float)
    return poisson_deviance(claims, np.asarray(pred_rate, dtype=float) * exposure) / float(exposure.sum())


def lorenz_curve(pred_rate, actual, exposure):
    """Cumulative exposure (x) against cumulative actual (y), policies ordered from lowest
    to highest predicted rate. Policies with the same prediction are pooled into one point."""
    d = pd.DataFrame({"p": np.asarray(pred_rate, dtype=float), "a": np.asarray(actual, dtype=float),
                      "e": np.asarray(exposure, dtype=float)})
    g = d.groupby("p", sort=True)[["e", "a"]].sum()
    x = np.concatenate(([0.0], np.cumsum(g["e"].to_numpy()) / g["e"].sum()))
    y = np.concatenate(([0.0], np.cumsum(g["a"].to_numpy()) / g["a"].sum()))
    return x, y


def gini(pred_rate, actual, exposure):
    """Gini = 1 - 2 * area under the Lorenz curve. Zero for a constant prediction."""
    x, y = lorenz_curve(pred_rate, actual, exposure)
    area = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
    return 1.0 - 2.0 * area


def actual_vs_expected(actual, pred_rate, exposure, by=None):
    """Actual over expected, overall (by=None) or for each level of `by`."""
    d = pd.DataFrame({"actual": np.asarray(actual, dtype=float),
                      "expected": np.asarray(pred_rate, dtype=float) * np.asarray(exposure, dtype=float),
                      "exposure": np.asarray(exposure, dtype=float)})
    if by is None:
        out = d.sum().to_frame().T
    else:
        out = d.groupby(np.asarray(by), observed=True).sum()
    out["a_over_e"] = out["actual"] / out["expected"]
    return out


def double_lift(pred_rate_a, pred_rate_b, actual, exposure, n_bins=10):
    """Double-lift table for models A and B.

    Policies are sorted by the ratio B / A and cut into bins of equal exposure. In each bin the
    table gives the actual rate and each model's predicted rate. The model whose line stays
    closer to the actual line across the bins is the better one where the two disagree.
    """
    d = pd.DataFrame({"a": np.asarray(pred_rate_a, dtype=float), "b": np.asarray(pred_rate_b, dtype=float),
                      "actual": np.asarray(actual, dtype=float), "exposure": np.asarray(exposure, dtype=float)})
    d["ratio"] = d["b"] / d["a"]
    d = d.sort_values("ratio", kind="mergesort")
    cum = (d["exposure"].cumsum() - d["exposure"] / 2.0) / d["exposure"].sum()
    d["bin"] = np.minimum((cum * n_bins).astype(int), n_bins - 1) + 1
    d["exp_a"] = d["a"] * d["exposure"]
    d["exp_b"] = d["b"] * d["exposure"]
    g = d.groupby("bin").agg(exposure=("exposure", "sum"), actual=("actual", "sum"), exp_a=("exp_a", "sum"),
                             exp_b=("exp_b", "sum"), ratio_min=("ratio", "min"), ratio_max=("ratio", "max"))
    return pd.DataFrame({
        "exposure": g["exposure"], "ratio_min": g["ratio_min"], "ratio_max": g["ratio_max"],
        "actual_rate": g["actual"] / g["exposure"], "rate_a": g["exp_a"] / g["exposure"], "rate_b": g["exp_b"] / g["exposure"],
    })


def poisson_deviance_terms(y, mu):
    """Each policy's contribution to the Poisson deviance."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    ylogy = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return 2.0 * (ylogy - (y - mu))


def bootstrap_deviance_difference(claims, pred_rate_a, pred_rate_b, exposure, n_boot=1000, seed=0):
    """Paired bootstrap of deviance(A) - deviance(B), per unit exposure.

    Policies are resampled with replacement and both models are scored on the same resample,
    so the interval reflects how much the difference depends on which policies are in the
    holdout. A positive value means model B is better. Returns (estimate, 2.5%, 97.5%).
    """
    exposure = np.asarray(exposure, dtype=float)
    diff = (poisson_deviance_terms(claims, np.asarray(pred_rate_a) * exposure)
            - poisson_deviance_terms(claims, np.asarray(pred_rate_b) * exposure))
    rng = np.random.default_rng(seed)
    n = len(diff)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[i] = diff[idx].sum() / exposure[idx].sum()
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(diff.sum() / exposure.sum()), float(low), float(high)


def frequency_summary(claims, pred_rate, exposure, null_rate):
    """The primary metrics for one frequency model on one dataset."""
    dev = mean_poisson_deviance(claims, pred_rate, exposure)
    null = mean_poisson_deviance(claims, np.full(len(np.asarray(claims)), null_rate), exposure)
    return {
        "poisson_deviance": dev,
        "null_deviance": null,
        "deviance_explained": 1.0 - dev / null,
        "gini": gini(pred_rate, claims, exposure),
        "a_over_e": float(np.sum(claims) / np.sum(np.asarray(pred_rate) * np.asarray(exposure))),
    }
