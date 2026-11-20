"""The from-scratch NumPy solver against statsmodels (the test oracle)."""

import numpy as np
import pandas as pd
import pytest

import data
from design import FINAL_FREQUENCY_TERMS, FREQUENCY_TERMS, SEVERITY_TERMS, Design, add_model_columns
from glm import fit_gamma, fit_poisson
from irls import fit_irls, irls_step

RTOL, ATOL = 1e-6, 1e-8


def assert_matches_statsmodels(ours, theirs, check_dispersion=False):
    np.testing.assert_allclose(ours["coef"], theirs.params, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(ours["std_err"], theirs.bse, rtol=RTOL, atol=ATOL)
    assert ours["deviance"] == pytest.approx(theirs.deviance, rel=1e-9)
    if check_dispersion:
        assert ours["dispersion"] == pytest.approx(theirs.scale, rel=1e-7)   # Pearson chi2 / df_resid
    assert ours["converged"]


@pytest.fixture(scope="module")
def poisson_data():
    rng = np.random.default_rng(0)
    n = 5000
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.integers(0, 2, n), rng.normal(size=n)])
    beta = np.array([-2.0, 0.3, 0.5, 0.0])       # the last coefficient is exactly zero
    exposure = rng.uniform(0.05, 1.0, n)
    y = rng.poisson(exposure * np.exp(X @ beta))
    return X, y, exposure, beta


@pytest.fixture(scope="module")
def gamma_data():
    rng = np.random.default_rng(1)
    n = 5000
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.integers(0, 2, n)])
    beta = np.array([7.0, -0.2, 0.4])
    weights = rng.integers(1, 4, n).astype(float)
    shape = 1.5
    y = rng.gamma(shape * weights, np.exp(X @ beta) / (shape * weights))
    return X, y, weights, beta


def test_poisson_with_offset_matches_statsmodels_and_recovers_the_truth(poisson_data):
    X, y, exposure, beta = poisson_data
    ours = fit_irls(X, y, family="poisson", offset=np.log(exposure))
    assert_matches_statsmodels(ours, fit_poisson(X, y, exposure))
    assert np.all(np.abs(ours["coef"] - beta) < 4 * ours["std_err"])
    assert ours["n_iter"] < 15


def test_gamma_with_prior_weights_matches_statsmodels_and_recovers_the_truth(gamma_data):
    X, y, weights, beta = gamma_data
    ours = fit_irls(X, y, family="gamma", weights=weights)
    assert_matches_statsmodels(ours, fit_gamma(X, y, weights), check_dispersion=True)
    assert np.all(np.abs(ours["coef"] - beta) < 4 * ours["std_err"])
    assert ours["dispersion"] == pytest.approx(1 / 1.5, rel=0.1)


def test_poisson_irls_step_is_a_newton_raphson_step(poisson_data):
    # Canonical link: gradient X'(y - mu), Hessian -X' diag(mu) X. Newton's update from any
    # beta equals the IRLS update from the same beta.
    X, y, exposure, _ = poisson_data
    beta = np.array([-1.0, 0.1, 0.1, 0.1])       # deliberately not the optimum
    mu = exposure * np.exp(X @ beta)
    gradient = X.T @ (y - mu)
    hessian = -(X.T * mu) @ X
    newton = beta - np.linalg.solve(hessian, gradient)
    irls = irls_step(X, y, beta, "poisson", np.log(exposure), np.ones(len(y)))
    np.testing.assert_allclose(irls, newton, rtol=1e-10, atol=1e-12)


def test_gamma_irls_step_is_fisher_scoring_not_newton_raphson(gamma_data):
    # Log link is not canonical for the Gamma family. Up to the dispersion, the log-likelihood is
    # sum w (-y / mu - log mu): gradient X' w (y - mu) / mu, observed Hessian -X' diag(w y / mu) X,
    # expected Hessian -X' diag(w) X. IRLS uses the expected one.
    X, y, weights, _ = gamma_data
    beta = np.array([6.5, 0.0, 0.0])             # deliberately not the optimum
    mu = np.exp(X @ beta)
    gradient = X.T @ (weights * (y - mu) / mu)
    fisher = beta + np.linalg.solve((X.T * weights) @ X, gradient)
    newton = beta + np.linalg.solve((X.T * (weights * y / mu)) @ X, gradient)
    irls = irls_step(X, y, beta, "gamma", np.zeros(len(y)), weights)
    np.testing.assert_allclose(irls, fisher, rtol=1e-10, atol=1e-12)
    assert np.max(np.abs(irls - newton)) > 1e-3
    # Both iterations still end at the same maximum likelihood estimate.
    converged = fit_irls(X, y, family="gamma", weights=weights)["coef"]
    b = beta.copy()
    for _ in range(50):
        m = np.exp(X @ b)
        b = b + np.linalg.solve((X.T * (weights * y / m)) @ X, X.T @ (weights * (y - m) / m))
    np.testing.assert_allclose(b, converged, rtol=1e-8)


def test_rejects_unknown_family(poisson_data):
    X, y, _, _ = poisson_data
    with pytest.raises(ValueError):
        fit_irls(X, y, family="binomial")


@pytest.fixture(scope="module")
def train():
    if not (data.CACHE_DIR / "freMTPL2_raw.pkl").exists():
        pytest.skip("raw data cache not present")
    policies, _, _ = data.load_dataset()
    return add_model_columns(policies[~policies["is_holdout"]])


@pytest.mark.parametrize("terms, width", [(FINAL_FREQUENCY_TERMS, 55), (FREQUENCY_TERMS, 63)], ids=["final", "nine_factor"])
def test_reproduces_the_pipeline_frequency_glm(train, terms, width):
    X = Design(terms).fit_transform(train).to_numpy()
    theirs = fit_poisson(X, train["ClaimNb"], train["Exposure"])
    ours = fit_irls(X, train["ClaimNb"], family="poisson", offset=np.log(train["Exposure"].to_numpy()))
    assert_matches_statsmodels(ours, theirs)
    assert X.shape[1] == width


def test_reproduces_the_pipeline_severity_glm(train):
    claims = train[train["ClaimRecords"] > 0]
    X = Design(SEVERITY_TERMS).fit_transform(claims).to_numpy()
    theirs = fit_gamma(X, claims["AvgSevCapped"], claims["ClaimRecords"])
    ours = fit_irls(X, claims["AvgSevCapped"], family="gamma", weights=claims["ClaimRecords"])
    assert_matches_statsmodels(ours, theirs, check_dispersion=True)
