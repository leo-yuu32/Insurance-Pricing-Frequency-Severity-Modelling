"""LightGBM claim frequency model.

Poisson objective. Exposure enters only as an offset, through `init_score`:

    init_score = log(Exposure) + log(base_rate)

where base_rate is the training claim frequency. LightGBM does not add the init_score back
when predicting, so the predicted rate per policy-year is base_rate * exp(raw_score) and the
predicted claim count is that rate times Exposure. Every validation Dataset is built by the
same function, so it always carries its own init_score.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import SEED
from metrics import poisson_deviance

FEATURES = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]
# The final model leaves out BonusMalus, which appears to be recorded after some of the claims it
# would predict (see README.md, Caveats). The nine-factor model is kept as a comparison.
FINAL_FEATURES = [f for f in FEATURES if f != "BonusMalus"]
# Brand and region are encoded as integers ordered by training claim frequency, so every split is
# numeric. SHAP interaction values do not support LightGBM's native categorical splits.
ORDERED_BY_FREQUENCY = ["VehBrand", "Region"]
# Used only when monotone=True. The primary model is unconstrained.
MONOTONE_INCREASING = ["BonusMalus"]
MAX_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 50

FIXED_PARAMS = {
    "objective": "poisson",
    "metric": "poisson",
    "verbosity": -1,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 8,
}


def category_levels(train):
    """For brand and region: level -> rank of its claim frequency in `train` (0 = lowest)."""
    levels = {}
    for col in ORDERED_BY_FREQUENCY:
        g = train.groupby(col)[["ClaimNb", "Exposure"]].sum()
        order = (g["ClaimNb"] / g["Exposure"]).sort_values(kind="mergesort").index
        levels[col] = {level: rank for rank, level in enumerate(order)}
    return levels


def make_features(df, levels, features=None):
    """The raw rating factors (default: all nine), all numeric. Area is ordinal (A to F), VehGas is a 0/1 flag."""
    X = pd.DataFrame(index=df.index)
    for col in features or FEATURES:
        if col == "Area":
            X[col] = df[col].map({a: i for i, a in enumerate("ABCDEF", start=1)}).astype(float)
        elif col == "VehGas":
            X[col] = (df[col] == "Diesel").astype(float)
        elif col in ORDERED_BY_FREQUENCY:
            X[col] = df[col].map(levels[col]).astype(float)  # an unseen level becomes missing
        else:
            X[col] = df[col].astype(float)
    return X


def make_dataset(df, levels, base_rate, features=None):
    return lgb.Dataset(
        make_features(df, levels, features), label=df["ClaimNb"].to_numpy(dtype=float),
        init_score=np.log(df["Exposure"].to_numpy(dtype=float) * base_rate), free_raw_data=False,
    )


def full_params(tuned, additive=False, monotone=False, features=None):
    features = features or FEATURES
    params = {**FIXED_PARAMS, **tuned}
    if monotone:
        params["monotone_constraints"] = [1 if f in MONOTONE_INCREASING else 0 for f in features]
    if additive:
        # Each tree may split on one feature only, so the model is a sum of one-factor effects.
        params["interaction_constraints"] = [[i] for i in range(len(features))]
    return params


def fit(train, tuned, n_rounds, additive=False, monotone=False, valid=None, features=None):
    """Fit on `train`. With `valid`, stop early on its Poisson loss and ignore n_rounds.
    `features` defaults to all nine rating factors."""
    features = list(features or FEATURES)
    levels = category_levels(train)
    base_rate = float(train["ClaimNb"].sum() / train["Exposure"].sum())
    dtrain = make_dataset(train, levels, base_rate, features)
    params = full_params(tuned, additive, monotone, features)
    if valid is None:
        booster = lgb.train(params, dtrain, num_boost_round=n_rounds)
    else:
        dvalid = make_dataset(valid, levels, base_rate, features)  # carries its own init_score
        booster = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid],
                            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    return {"booster": booster, "levels": levels, "base_rate": base_rate, "features": features}


def predict_rate(model, df):
    """Predicted claims per policy-year. Multiply by Exposure for a claim count."""
    raw = model["booster"].predict(make_features(df, model["levels"], model["features"]), raw_score=True)
    return model["base_rate"] * np.exp(raw)


def cv_score(train, tuned, additive=False, monotone=False, features=None):
    """Out-of-fold Poisson deviance per unit exposure, and the mean early-stopped number of trees."""
    total, rounds = 0.0, []
    for k in sorted(train["fold"].unique()):
        fit_rows, test_rows = train[train["fold"] != k], train[train["fold"] == k]
        model = fit(fit_rows, tuned, None, additive, monotone, valid=test_rows, features=features)
        rate = predict_rate(model, test_rows)
        total += poisson_deviance(test_rows["ClaimNb"], rate * test_rows["Exposure"])
        rounds.append(model["booster"].best_iteration)
    return total / float(train["Exposure"].sum()), int(round(float(np.mean(rounds))))
