"""A GLM solver written from scratch in NumPy: iteratively reweighted least squares (IRLS).

Supports the two models the pipeline uses, both with a log link:
    family="poisson"   claim counts, with an offset (log exposure)
    family="gamma"     claim severity, with prior weights (number of claims)

This is not the pipeline's engine (statsmodels is). It is an independent implementation, and
tests/test_irls.py checks that it reproduces the statsmodels fit on the pipeline's own design
matrices.

The algorithm, for a log link (eta = X beta + offset, mu = exp(eta), so d mu / d eta = mu):

    working response   z = (eta - offset) + (y - mu) / mu
    working weights    W = w * mu      for Poisson   (variance function V(mu) = mu)
                       W = w           for Gamma     (V(mu) = mu^2, so mu^2 / V(mu) = 1)
    update             beta = (X' W X)^-1 X' W z

With the Poisson family the log link is the canonical link, the observed and expected information
are the same matrix, and this update is exactly a Newton-Raphson step. With the Gamma family the
log link is not canonical: the update uses the expected information, so it is Fisher scoring, and
it differs from Newton-Raphson away from the optimum (tests/test_irls.py shows both facts).
"""

import numpy as np

FAMILIES = ("poisson", "gamma")


def _deviance(y, mu, weights, family):
    if family == "poisson":
        ylogy = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
        return float(2.0 * np.sum(weights * (ylogy - (y - mu))))
    return float(2.0 * np.sum(weights * ((y - mu) / mu - np.log(y / mu))))


def _working_weights(mu, weights, family):
    return weights * mu if family == "poisson" else weights


def irls_step(X, y, beta, family, offset, weights):
    """One IRLS update from `beta`."""
    eta = X @ beta + offset
    mu = np.exp(eta)
    z = (eta - offset) + (y - mu) / mu
    W = _working_weights(mu, weights, family)
    XtW = X.T * W
    return np.linalg.solve(XtW @ X, XtW @ z)


def fit_irls(X, y, family="poisson", offset=None, weights=None, tol=1e-10, max_iter=100):
    """Fit a log-link GLM. Returns a dict with the coefficients and the usual summary quantities."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)

    # Start from fitted values halfway between each observation and the overall mean, so that
    # log(mu) is defined when y = 0. The first update is a weighted regression of z on X.
    mu = (y + np.average(y, weights=weights)) / 2.0
    eta = np.log(mu)
    deviance = _deviance(y, mu, weights, family)
    converged = False
    for iteration in range(1, max_iter + 1):
        z = (eta - offset) + (y - mu) / mu
        W = _working_weights(mu, weights, family)
        XtW = X.T * W
        beta = np.linalg.solve(XtW @ X, XtW @ z)
        eta = X @ beta + offset
        mu = np.exp(eta)
        previous, deviance = deviance, _deviance(y, mu, weights, family)
        if abs(deviance - previous) <= tol * (abs(deviance) + 0.1):
            converged = True
            break

    # Dispersion: 1 for Poisson. For Gamma, the Pearson chi-squared statistic over the residual
    # degrees of freedom (the same convention as statsmodels' default scale).
    if family == "poisson":
        dispersion = 1.0
    else:
        dispersion = float(np.sum(weights * (y - mu) ** 2 / mu**2) / (n - p))
    information = (X.T * _working_weights(mu, weights, family)) @ X
    covariance = dispersion * np.linalg.inv(information)
    return {
        "coef": beta, "std_err": np.sqrt(np.diag(covariance)), "covariance": covariance,
        "deviance": deviance, "dispersion": dispersion, "n_iter": iteration, "converged": converged, "mu": mu,
    }
