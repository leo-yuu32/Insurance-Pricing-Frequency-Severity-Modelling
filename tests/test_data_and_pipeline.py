"""Tests on the real dataset and on the pipeline's saved predictions.

They are skipped when the local data cache or the predictions file is missing, so they never
trigger a download or a model fit. Run `python src/run_glm.py` first to create both.
"""

import numpy as np
import pandas as pd
import pytest

import data
from design import add_model_columns

PREDICTIONS = data.ROOT / "outputs" / "predictions.parquet"


@pytest.fixture(scope="module")
def dataset():
    if not (data.CACHE_DIR / "freMTPL2_raw.pkl").exists():
        pytest.skip("raw data cache not present")
    return data.load_dataset()


@pytest.fixture(scope="module")
def predictions():
    if not PREDICTIONS.exists():
        pytest.skip("outputs/predictions.parquet not present")
    return add_model_columns(pd.read_parquet(PREDICTIONS))


def test_cleaning_rules(dataset):
    policies, _, _ = dataset
    assert policies["IDpol"].min() > data.ANOMALOUS_BLOCK_MAX_IDPOL
    assert policies["IDpol"].is_unique
    assert policies["Exposure"].between(0, 1, inclusive="right").all()
    assert (policies["ClaimNb"] == policies["ClaimRecords"].clip(upper=data.CLAIM_COUNT_CAP)).all()
    assert policies[data.RATING_FACTORS].notna().all().all()


def test_claims_reconcile_with_policies(dataset):
    policies, claims, _ = dataset
    assert len(claims) == policies["ClaimRecords"].sum()
    assert claims["ClaimAmount"].sum() == pytest.approx(policies["TotalLoss"].sum())
    assert claims["ClaimAmountCapped"].sum() == pytest.approx(policies["TotalLossCapped"].sum())
    assert (policies.loc[policies["ClaimRecords"] == 0, "TotalLoss"] == 0).all()


def test_large_loss_cap_uses_training_claims_only(dataset):
    _, claims, cap = dataset
    train_amounts = claims.loc[~claims["is_holdout"], "ClaimAmount"]
    assert cap == pytest.approx(train_amounts.quantile(data.LARGE_LOSS_QUANTILE))
    assert claims["ClaimAmountCapped"].max() == pytest.approx(cap)


def test_split_is_grouped_and_stratified(dataset):
    policies, _, _ = dataset
    assert set(policies["fold"]) == set(range(data.N_FOLDS))
    assert (policies["is_holdout"] == (policies["fold"] == 0)).all()
    assert policies.groupby("Profile")["fold"].nunique().max() == 1      # no profile on both sides
    assert policies["is_holdout"].mean() == pytest.approx(0.2, abs=0.001)
    claim_rate = (policies["ClaimNb"] > 0).groupby(policies["fold"]).mean()
    assert claim_rate.max() - claim_rate.min() < 0.0005                  # stratified on the claim indicator


def test_holdout_matches_the_saved_ids(dataset):
    policies, _, _ = dataset
    if not data.HOLDOUT_IDS_PATH.exists():
        pytest.skip("artifacts/holdout_idpol.txt not present")
    current = np.sort(policies.loc[policies["is_holdout"], "IDpol"].to_numpy())
    np.testing.assert_array_equal(current, data.read_holdout_ids())


def test_frequency_glm_balances_on_training_data(predictions):
    train = predictions[~predictions["is_holdout"]]
    expected = train["freq_glm"] * train["Exposure"]
    assert expected.sum() == pytest.approx(train["ClaimNb"].sum(), rel=1e-6)
    for factor in ["DrivAgeBand", "VehAgeBand", "VehPowerBand", "BonusMalusBand", "VehBrand", "VehGas", "Region"]:
        by_level = pd.DataFrame({"e": expected, "a": train["ClaimNb"]}).groupby(train[factor]).sum()
        np.testing.assert_allclose(by_level["e"], by_level["a"], rtol=1e-5, err_msg=factor)


def test_pure_premium_is_rebalanced_to_capped_training_losses(predictions):
    train = predictions[~predictions["is_holdout"]]
    assert (train["pp_glm"] * train["Exposure"]).sum() == pytest.approx(train["TotalLossCapped"].sum(), rel=1e-9)


def test_predictions_are_positive_and_complete(predictions):
    for col in [c for c in predictions.columns if c.startswith(("freq_", "sev_", "pp_"))]:
        assert predictions[col].notna().all() and (predictions[col] > 0).all(), col
