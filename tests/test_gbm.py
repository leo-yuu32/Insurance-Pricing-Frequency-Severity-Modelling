import numpy as np
import pandas as pd
import pytest

import gbm
from metrics import mean_poisson_deviance

PARAMS = {"learning_rate": 0.1, "num_leaves": 16, "min_data_in_leaf": 50, "lambda_l2": 1.0}


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n = 30000
    df = pd.DataFrame({
        "Area": rng.choice(list("ABCDEF"), n),
        "VehPower": rng.integers(4, 16, n).astype(float),
        "VehAge": rng.integers(0, 25, n).astype(float),
        "DrivAge": rng.integers(18, 90, n).astype(float),
        "BonusMalus": rng.choice([50.0, 60.0, 80.0, 100.0, 125.0], n, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        "VehBrand": rng.choice(["B1", "B2", "B12"], n),
        "VehGas": rng.choice(["Diesel", "Regular"], n),
        "Density": np.exp(rng.uniform(0, 10, n)),
        "Region": rng.choice(["R11", "R24", "R82", "R93"], n),
        "Exposure": rng.choice([0.05, 0.25, 0.5, 1.0], n),
        "fold": rng.integers(1, 5, n),
    })
    young_malus = (df["DrivAge"] < 30) & (df["BonusMalus"] >= 100)
    rate = 0.08 * np.exp(0.012 * (df["BonusMalus"] - 50) + 0.4 * (df["DrivAge"] < 25) + 0.5 * young_malus + 0.2 * (df["VehBrand"] == "B12"))
    df["ClaimNb"] = rng.poisson(rate * df["Exposure"])
    return df


@pytest.fixture(scope="module")
def model(toy):
    return gbm.fit(toy, PARAMS, n_rounds=150)


def test_predicted_claims_match_the_observed_total_on_training_data(toy, model):
    # Trap 1: predict() does not add the init_score back. If the offset were dropped or
    # double-counted, this total would be far off.
    predicted = gbm.predict_rate(model, toy) * toy["Exposure"]
    assert predicted.sum() == pytest.approx(toy["ClaimNb"].sum(), rel=0.01)


def test_predicted_rate_is_per_policy_year_and_ignores_exposure(toy, model):
    short = toy.head(500).assign(Exposure=0.05)
    full = toy.head(500).assign(Exposure=1.0)
    np.testing.assert_allclose(gbm.predict_rate(model, short), gbm.predict_rate(model, full))
    assert "Exposure" not in gbm.make_features(toy, model["levels"]).columns
    by_exposure = pd.DataFrame({"rate": gbm.predict_rate(model, toy), "e": toy["Exposure"]}).groupby("e")["rate"].mean()
    assert by_exposure.max() / by_exposure.min() < 1.05  # the true rate does not depend on exposure


def test_every_dataset_carries_its_own_init_score(toy):
    # Trap 2: a validation set without an init_score is scored as if every policy had one year.
    valid = toy[toy["fold"] == 1]
    dataset = gbm.make_dataset(valid, gbm.category_levels(toy), base_rate=0.1)
    np.testing.assert_allclose(dataset.get_init_score(), np.log(valid["Exposure"].to_numpy() * 0.1))


def test_early_stopping_and_cross_validation(toy):
    deviance, n_rounds = gbm.cv_score(toy, PARAMS)
    assert 1 <= n_rounds < gbm.MAX_ROUNDS
    null_rate = toy["ClaimNb"].sum() / toy["Exposure"].sum()
    null = mean_poisson_deviance(toy["ClaimNb"], np.full(len(toy), null_rate), toy["Exposure"])
    assert deviance < null


def test_final_features_leave_out_bonus_malus(toy):
    assert gbm.FINAL_FEATURES == [f for f in gbm.FEATURES if f != "BonusMalus"]
    model = gbm.fit(toy, PARAMS, n_rounds=50, features=gbm.FINAL_FEATURES)
    assert model["booster"].feature_name() == gbm.FINAL_FEATURES
    low, high = toy.head(500).assign(BonusMalus=50.0), toy.head(500).assign(BonusMalus=230.0)
    np.testing.assert_array_equal(gbm.predict_rate(model, low), gbm.predict_rate(model, high))
    deviance, _ = gbm.cv_score(toy, PARAMS, features=gbm.FINAL_FEATURES)
    assert deviance > gbm.cv_score(toy, PARAMS)[0]   # the toy rate does depend on BonusMalus


def test_monotone_option_makes_the_bonus_malus_effect_increasing(toy):
    model = gbm.fit(toy, PARAMS, n_rounds=150, monotone=True)
    grid = pd.concat([toy.head(200).assign(BonusMalus=float(b)) for b in range(50, 231, 5)], keys=range(50, 231, 5))
    rate = pd.Series(gbm.predict_rate(model, grid), index=grid.index).unstack(level=0)
    assert (rate.diff(axis=1).iloc[:, 1:] >= -1e-12).all().all()


def test_additive_model_has_no_interactions(toy):
    additive = gbm.fit(toy, PARAMS, n_rounds=150, additive=True)
    base = toy.head(300)
    a = base.assign(DrivAge=20.0, BonusMalus=50.0)
    b = base.assign(DrivAge=20.0, BonusMalus=125.0)
    c = base.assign(DrivAge=60.0, BonusMalus=50.0)
    d = base.assign(DrivAge=60.0, BonusMalus=125.0)

    def log_rate(m, rows):
        return np.log(gbm.predict_rate(m, rows))

    # Additive on the log scale: the effect of BonusMalus is the same at every driver age.
    np.testing.assert_allclose(log_rate(additive, b) - log_rate(additive, a), log_rate(additive, d) - log_rate(additive, c), atol=1e-9)
    # The unconstrained model is free to learn the interaction that is in the data.
    full = gbm.fit(toy, PARAMS, n_rounds=150)
    gap = (log_rate(full, b) - log_rate(full, a)) - (log_rate(full, d) - log_rate(full, c))
    assert np.abs(gap).mean() > 0.05
