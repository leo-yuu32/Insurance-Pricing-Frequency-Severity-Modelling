"""The pipeline's GLMs, fitted with statsmodels by unpenalised maximum likelihood.

Frequency: Poisson, log link, log(exposure) as an offset, response = claim count.
Severity:  Gamma, log link, response = average capped claim amount per policy,
           variance weights = number of claims on the policy.
"""

import numpy as np
import statsmodels.api as sm

from design import Design
from metrics import poisson_deviance


def fit_poisson(X, claims, exposure):
    model = sm.GLM(np.asarray(claims, dtype=float), np.asarray(X, dtype=float),
                   family=sm.families.Poisson(), offset=np.log(np.asarray(exposure, dtype=float)))
    return model.fit()


def fit_gamma(X, avg_severity, n_claims):
    model = sm.GLM(np.asarray(avg_severity, dtype=float), np.asarray(X, dtype=float),
                   family=sm.families.Gamma(link=sm.families.links.Log()),
                   var_weights=np.asarray(n_claims, dtype=float))
    return model.fit()


def predict(result, X):
    """exp(X beta): claims per policy-year for the frequency model, euros per claim for severity."""
    return np.exp(np.asarray(X, dtype=float) @ np.asarray(result.params))


def gamma_deviance(y, mu, weights):
    y, mu, weights = (np.asarray(v, dtype=float) for v in (y, mu, weights))
    return float(2.0 * np.sum(weights * ((y - mu) / mu - np.log(y / mu))))


def cv_frequency_deviance(train, terms, interactions=()):
    """Mean out-of-fold Poisson deviance per unit exposure over the folds in train['fold']."""
    total = 0.0
    for k in sorted(train["fold"].unique()):
        fit_rows, test_rows = train[train["fold"] != k], train[train["fold"] == k]
        design = Design(terms, interactions).fit(fit_rows)
        result = fit_poisson(design.transform(fit_rows), fit_rows["ClaimNb"], fit_rows["Exposure"])
        rate = predict(result, design.transform(test_rows))
        total += poisson_deviance(test_rows["ClaimNb"], rate * test_rows["Exposure"])
    return total / float(train["Exposure"].sum())


def cv_severity_deviance(train_claims, terms):
    """Mean out-of-fold Gamma deviance per claim. `terms=[]` is the intercept-only model."""
    total = 0.0
    for k in sorted(train_claims["fold"].unique()):
        fit_rows, test_rows = train_claims[train_claims["fold"] != k], train_claims[train_claims["fold"] == k]
        design = Design(terms).fit(fit_rows)
        result = fit_gamma(design.transform(fit_rows), fit_rows["AvgSevCapped"], fit_rows["ClaimRecords"])
        mu = predict(result, design.transform(test_rows))
        total += gamma_deviance(test_rows["AvgSevCapped"], mu, test_rows["ClaimRecords"])
    return total / float(train_claims["ClaimRecords"].sum())
