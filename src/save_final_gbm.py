"""Refit the final GBM once and save it for the dashboard's pricing tool.

The final GBM was fitted by run_final.py but not saved. This script refits it with the committed
specification (the tuned parameters in outputs/gbm_tuning_final.json, seed 42, deterministic
LightGBM), checks that its predictions match the saved predictions, and only then saves it.
Nothing is rescored and nothing is chosen.

Run from the repo root, after run_final.py:
    python src/save_final_gbm.py

Reads  outputs/predictions.parquet, outputs/gbm_tuning_final.json, outputs/metrics_final.json.
Writes artifacts/gbm_final.txt    the LightGBM model (text format)
       artifacts/gbm_final.json   what is needed to use it: features, encodings, base rate, factors
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import gbm
from run_glm import OUTPUTS, PREDICTIONS_PATH, rebalance_factor

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "artifacts" / "gbm_final.txt"
INFO_PATH = ROOT / "artifacts" / "gbm_final.json"
RTOL = 1e-9  # relative tolerance for "the same predictions"


def compare(new, saved):
    """Largest relative difference between two sets of predicted rates, and whether all are within RTOL."""
    rel = np.abs(np.asarray(new) / np.asarray(saved) - 1.0)
    return float(rel.max()), bool(np.allclose(new, saved, rtol=RTOL, atol=0.0))


def main():
    tuning = json.loads((OUTPUTS / "gbm_tuning_final.json").read_text())
    final = json.loads((OUTPUTS / "metrics_final.json").read_text())["gbm_final"]
    best = tuning["best"]
    assert tuning["features"] == gbm.FINAL_FEATURES == final["features"]
    assert best["params"] == final["params"] and best["n_rounds"] == final["n_rounds"]

    policies = pd.read_parquet(PREDICTIONS_PATH)
    train = policies[~policies["is_holdout"]]
    model = gbm.fit(train, best["params"], best["n_rounds"], features=gbm.FINAL_FEATURES)
    raw = gbm.predict_rate(model, policies)
    freq_factor = rebalance_factor(train["ClaimNb"], raw[~policies["is_holdout"].to_numpy()], train["Exposure"])
    rate = raw * freq_factor

    holdout = policies["is_holdout"].to_numpy()
    worst_holdout, ok_holdout = compare(rate[holdout], policies.loc[holdout, "freq_gbm_final"])
    worst_train, ok_train = compare(rate[~holdout], policies.loc[~holdout, "freq_gbm_final"])
    print(f"holdout:  largest relative difference {worst_holdout:.2e}  within {RTOL:g}: {ok_holdout}")
    print(f"training: largest relative difference {worst_train:.2e}  within {RTOL:g}: {ok_train}")
    print(f"frequency rebalancing factor: refit {freq_factor!r}, saved {final['frequency_rebalance_factor']!r}")
    if not (ok_holdout and ok_train):
        raise SystemExit("The refit does not reproduce the saved predictions. Nothing was saved.")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model["booster"].save_model(str(MODEL_PATH))
    info = {
        "description": "Final GBM (LightGBM, Poisson), refitted with the committed specification. Predicted claims "
                       "per policy-year = base_rate * exp(raw_score) * frequency_rebalance_factor.",
        "features": model["features"],
        "encoding": {"Area": "A to F as 1 to 6", "VehGas": "1 for Diesel, 0 otherwise",
                     "VehBrand": "rank of training claim frequency", "Region": "rank of training claim frequency"},
        "levels": model["levels"],
        "base_rate": model["base_rate"],
        "frequency_rebalance_factor": final["frequency_rebalance_factor"],
        "pure_premium_rebalance_factor": final["pure_premium_rebalance_factor"],
        "params": best["params"], "n_rounds": best["n_rounds"],
    }
    INFO_PATH.write_text(json.dumps(info, indent=1), encoding="utf-8")

    # The dashboard loads the saved file, so check that path too.
    loaded = lgb.Booster(model_file=str(MODEL_PATH))
    X = gbm.make_features(policies[holdout], model["levels"], model["features"])
    reloaded = model["base_rate"] * np.exp(loaded.predict(X, raw_score=True)) * final["frequency_rebalance_factor"]
    worst_reloaded, ok_reloaded = compare(reloaded, policies.loc[holdout, "freq_gbm_final"])
    print(f"saved file, holdout: largest relative difference {worst_reloaded:.2e}  within {RTOL:g}: {ok_reloaded}")
    for path in (MODEL_PATH, INFO_PATH):
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.1f} KB")
    if not ok_reloaded:
        raise SystemExit("The saved file does not reproduce the saved predictions.")


if __name__ == "__main__":
    main()
