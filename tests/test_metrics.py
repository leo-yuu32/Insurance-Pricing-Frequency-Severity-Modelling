import numpy as np
import pytest
from sklearn.metrics import mean_poisson_deviance as sklearn_mean_poisson_deviance

import metrics


def test_gini_of_a_constant_prediction_is_zero():
    rng = np.random.default_rng(0)
    exposure = rng.uniform(0.01, 1.0, 500)
    actual = rng.poisson(0.2, 500)
    assert metrics.gini(np.full(500, 0.1), actual, exposure) == pytest.approx(0.0, abs=1e-12)


def test_gini_does_not_reward_ordering_by_exposure():
    # The legacy function sorted by prediction / exposure. Here all the claims sit on the
    # short-exposure policies, so that bug would give a large Gini to a model with no information.
    exposure = np.array([0.01, 0.01, 1.0, 1.0])
    actual = np.array([1.0, 1.0, 0.0, 0.0])
    assert metrics.gini(np.ones(4), actual, exposure) == pytest.approx(0.0, abs=1e-12)


def test_gini_known_values():
    exposure = np.ones(4)
    actual = np.array([0.0, 0.0, 0.0, 1.0])
    assert metrics.gini(np.array([1.0, 2.0, 3.0, 4.0]), actual, exposure) == pytest.approx(0.75)
    assert metrics.gini(np.array([4.0, 3.0, 2.0, 1.0]), actual, exposure) == pytest.approx(-0.75)


def test_gini_depends_only_on_the_ordering():
    rng = np.random.default_rng(1)
    pred = rng.uniform(0.02, 0.5, 300)
    exposure = rng.uniform(0.1, 1.0, 300)
    actual = rng.poisson(pred * exposure)
    base = metrics.gini(pred, actual, exposure)
    assert metrics.gini(3.0 * pred, actual, exposure) == pytest.approx(base)
    assert metrics.gini(np.log(pred), actual, exposure) == pytest.approx(base)
    assert metrics.gini(pred, actual, 2.0 * exposure) == pytest.approx(base)


def test_gini_pools_tied_predictions():
    # Two tied policies must give the same answer whatever order they arrive in.
    pred = np.array([0.1, 0.2, 0.2, 0.3])
    exposure = np.array([1.0, 0.5, 1.0, 1.0])
    actual = np.array([0.0, 1.0, 0.0, 2.0])
    swapped = [0, 2, 1, 3]
    assert metrics.gini(pred, actual, exposure) == pytest.approx(metrics.gini(pred[swapped], actual[swapped], exposure[swapped]))


def test_poisson_deviance_by_hand():
    assert metrics.poisson_deviance([2.0, 1.0], [2.0, 1.0]) == pytest.approx(0.0)
    assert metrics.poisson_deviance([0.0], [0.3]) == pytest.approx(0.6)
    assert metrics.poisson_deviance([1.0], [0.5]) == pytest.approx(2 * (np.log(2.0) - 0.5))


def test_mean_poisson_deviance_matches_sklearn():
    rng = np.random.default_rng(2)
    exposure = rng.uniform(0.05, 1.0, 2000)
    rate = rng.uniform(0.03, 0.3, 2000)
    claims = rng.poisson(rate * exposure)
    pred = rate * rng.uniform(0.7, 1.3, 2000)
    expected = sklearn_mean_poisson_deviance(claims / exposure, pred, sample_weight=exposure)
    assert metrics.mean_poisson_deviance(claims, pred, exposure) == pytest.approx(expected, rel=1e-10)


def test_actual_vs_expected():
    actual = np.array([1.0, 0.0, 2.0, 1.0])
    rate = np.array([0.5, 0.5, 1.0, 1.0])
    exposure = np.array([1.0, 1.0, 2.0, 2.0])
    overall = metrics.actual_vs_expected(actual, rate, exposure)
    assert overall["a_over_e"].iloc[0] == pytest.approx(4.0 / 5.0)
    by = metrics.actual_vs_expected(actual, rate, exposure, by=np.array(["a", "a", "b", "b"]))
    assert by.loc["a", "a_over_e"] == pytest.approx(1.0)
    assert by.loc["b", "a_over_e"] == pytest.approx(0.75)


def test_double_lift_bins_and_totals():
    rng = np.random.default_rng(3)
    n = 5000
    exposure = rng.uniform(0.1, 1.0, n)
    true_rate = rng.uniform(0.05, 0.3, n)
    actual = rng.poisson(true_rate * exposure)
    model_a = np.full(n, np.sum(true_rate * exposure) / exposure.sum())  # knows only the average
    model_b = true_rate                                                   # knows the truth
    table = metrics.double_lift(model_a, model_b, actual, exposure, n_bins=10)
    assert len(table) == 10
    assert table["exposure"].sum() == pytest.approx(exposure.sum())
    assert table["exposure"].max() / table["exposure"].min() < 1.05
    assert (table["actual_rate"] * table["exposure"]).sum() == pytest.approx(actual.sum())
    assert table["ratio_min"].is_monotonic_increasing
    # The informed model tracks the actual rate across the bins; the flat model cannot.
    error_a = np.abs(table["rate_a"] - table["actual_rate"]).mean()
    error_b = np.abs(table["rate_b"] - table["actual_rate"]).mean()
    assert error_b < error_a


def test_bootstrap_deviance_difference():
    rng = np.random.default_rng(4)
    n = 20000
    exposure = rng.uniform(0.1, 1.0, n)
    true_rate = rng.uniform(0.05, 0.4, n)
    claims = rng.poisson(true_rate * exposure)
    flat = np.full(n, claims.sum() / exposure.sum())
    est, low, high = metrics.bootstrap_deviance_difference(claims, flat, true_rate, exposure, n_boot=200, seed=0)
    direct = metrics.mean_poisson_deviance(claims, flat, exposure) - metrics.mean_poisson_deviance(claims, true_rate, exposure)
    assert est == pytest.approx(direct)
    assert 0 < low < est < high          # the true model is better, and the interval excludes zero
    same = metrics.bootstrap_deviance_difference(claims, flat, flat, exposure, n_boot=50, seed=0)
    assert same == (0.0, 0.0, 0.0)


def test_frequency_summary_of_the_null_model():
    exposure = np.array([1.0, 0.5, 1.0, 0.5])
    claims = np.array([0, 1, 0, 0])
    rate = claims.sum() / exposure.sum()
    out = metrics.frequency_summary(claims, np.full(4, rate), exposure, rate)
    assert out["deviance_explained"] == pytest.approx(0.0, abs=1e-12)
    assert out["a_over_e"] == pytest.approx(1.0)
    assert out["gini"] == pytest.approx(0.0, abs=1e-12)
