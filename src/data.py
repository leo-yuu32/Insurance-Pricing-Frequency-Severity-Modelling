"""Load freMTPL2, apply the documented cleaning and make the train/holdout split.

Every model in the rebuilt pipeline gets its data from `load_dataset()`.
The cleaning steps are summarised in README.md (Method).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedGroupKFold

from config import OPENML_FREQUENCY_DATA_ID, OPENML_SEVERITY_DATA_ID, SEED

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data"  # git-ignored
HOLDOUT_IDS_PATH = ROOT / "artifacts" / "holdout_idpol.txt"

# Every policy with IDpol up to this value has a claim, and almost none has a claim amount.
ANOMALOUS_BLOCK_MAX_IDPOL = 24951
EXPOSURE_CAP = 1.0
CLAIM_COUNT_CAP = 4
LARGE_LOSS_QUANTILE = 0.999
N_FOLDS = 5  # fold 0 is the holdout, folds 1 to 4 are the cross-validation folds

RATING_FACTORS = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]
CATEGORICAL_FACTORS = ["Area", "VehBrand", "VehGas", "Region"]
NUMERIC_FACTORS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]


def load_raw():
    """Return the raw frequency and severity tables, cached locally after the first download."""
    cache = CACHE_DIR / "freMTPL2_raw.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    freq = fetch_openml(data_id=OPENML_FREQUENCY_DATA_ID, as_frame=True, parser="auto").frame
    sev = fetch_openml(data_id=OPENML_SEVERITY_DATA_ID, as_frame=True, parser="auto").frame
    CACHE_DIR.mkdir(exist_ok=True)
    cache.write_bytes(pickle.dumps((freq, sev)))
    return freq, sev


def clean_policies(freq, sev):
    """One row per policy, with the claim count taken from the severity records."""
    df = freq.copy()
    df["IDpol"] = df["IDpol"].astype(int)
    for col in CATEGORICAL_FACTORS:
        df[col] = df[col].astype(str).str.strip("'")
    for col in NUMERIC_FACTORS:
        df[col] = pd.to_numeric(df[col]).astype(float)
    df["ClaimNbFile"] = pd.to_numeric(df["ClaimNb"]).astype(int)

    df = df[df["IDpol"] > ANOMALOUS_BLOCK_MAX_IDPOL].copy()

    sev_ids = sev["IDpol"].astype(int)
    n_records = sev_ids.value_counts()
    df["ClaimRecords"] = df["IDpol"].map(n_records).fillna(0).astype(int)
    df["ClaimNb"] = df["ClaimRecords"].clip(upper=CLAIM_COUNT_CAP)
    df["Exposure"] = pd.to_numeric(df["Exposure"]).clip(upper=EXPOSURE_CAP)

    # Policies with identical rating factors share a profile; the split keeps a profile on one side.
    df["Profile"] = df.groupby(RATING_FACTORS, sort=False).ngroup()
    return df.reset_index(drop=True)


def add_split(df):
    """Add `fold` and `is_holdout`.

    The policies are cut into N_FOLDS equal folds, stratified on whether the policy has a claim
    and never separating policies with the same profile. Fold 0 is the holdout (20%). Folds 1
    to 4 are the cross-validation folds used for every choice made on the training data.

    The assignment takes a few minutes to compute, so it is cached in the git-ignored data folder.
    """
    cache = CACHE_DIR / f"folds_seed{SEED}_n{N_FOLDS}_block{ANOMALOUS_BLOCK_MAX_IDPOL}.pkl"
    df = df.copy()
    if cache.exists():
        fold = df["IDpol"].map(pickle.loads(cache.read_bytes()))
        if fold.isna().any():
            raise ValueError(f"{cache} does not cover the current policies; delete it to recompute the split.")
        df["fold"] = fold.astype(int)
    else:
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        has_claim = (df["ClaimNb"] > 0).to_numpy()
        fold = np.empty(len(df), dtype=int)
        for k, (_, idx) in enumerate(splitter.split(df, has_claim, groups=df["Profile"])):
            fold[idx] = k
        df["fold"] = fold
        CACHE_DIR.mkdir(exist_ok=True)
        cache.write_bytes(pickle.dumps(df.set_index("IDpol")["fold"]))
    df["is_holdout"] = df["fold"] == 0
    return df


def build_claims(sev, policies):
    """One row per claim on a kept policy, capped at a quantile of the TRAINING claims only."""
    claims = pd.DataFrame({"IDpol": sev["IDpol"].astype(int), "ClaimAmount": pd.to_numeric(sev["ClaimAmount"])})
    claims = claims.merge(policies[["IDpol", "is_holdout"]], on="IDpol", how="inner")
    cap = float(claims.loc[~claims["is_holdout"], "ClaimAmount"].quantile(LARGE_LOSS_QUANTILE))
    claims["ClaimAmountCapped"] = claims["ClaimAmount"].clip(upper=cap)
    return claims, cap


def load_dataset():
    """Return (policies, claims, large_loss_cap).

    policies: one row per policy with rating factors, Exposure, ClaimNb, is_holdout,
              TotalLoss (uncapped), TotalLossCapped and AvgSevCapped.
    claims:   one row per claim with ClaimAmount and ClaimAmountCapped.
    """
    freq, sev = load_raw()
    policies = add_split(clean_policies(freq, sev))
    claims, cap = build_claims(sev, policies)
    totals = claims.groupby("IDpol").agg(TotalLoss=("ClaimAmount", "sum"), TotalLossCapped=("ClaimAmountCapped", "sum"))
    policies = policies.join(totals, on="IDpol")
    policies[["TotalLoss", "TotalLossCapped"]] = policies[["TotalLoss", "TotalLossCapped"]].fillna(0.0)
    policies["AvgSevCapped"] = np.where(
        policies["ClaimRecords"] > 0, policies["TotalLossCapped"] / policies["ClaimRecords"].clip(lower=1), np.nan
    )
    return policies, claims, cap


def save_holdout_ids(policies, path=HOLDOUT_IDS_PATH):
    ids = np.sort(policies.loc[policies["is_holdout"], "IDpol"].to_numpy())
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(str(i) for i in ids) + "\n")


def read_holdout_ids(path=HOLDOUT_IDS_PATH):
    return np.array([int(line) for line in path.read_text().split()])
