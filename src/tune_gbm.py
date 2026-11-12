"""Tune the LightGBM frequency model on the TRAINING data only.

Random search, scored by 4-fold cross-validated Poisson deviance with early stopping. The folds
are the grouped, stratified folds from data.py. The holdout is never loaded into a model here.

Run from the repo root:
    python src/tune_gbm.py             the full GBM
    python src/tune_gbm.py additive    the additive GBM (no interactions)
    python src/tune_gbm.py monotone    the GBM with BonusMalus constrained to be increasing

Writes outputs/gbm_tuning.json (or gbm_tuning_additive.json): every trial, and the chosen
parameters under "best". run_gbm.py reads the chosen parameters from that file.
"""

import json
import sys
import time

import numpy as np

import data
import gbm
from config import SEED
from run_glm import OUTPUTS

N_TRIALS = 30


def sample_params(rng):
    return {
        "learning_rate": round(float(10 ** rng.uniform(np.log10(0.03), np.log10(0.2))), 4),
        "num_leaves": int(round(2 ** rng.uniform(3, 6))),            # 8 to 64
        "min_data_in_leaf": int(round(10 ** rng.uniform(1.7, 3.3))),  # 50 to 2000
        "lambda_l2": round(float(10 ** rng.uniform(-2, 1.5)), 4),     # 0.01 to 32
    }


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else ""
    additive, monotone = variant == "additive", variant == "monotone"
    out_path = OUTPUTS / (f"gbm_tuning_{variant}.json" if variant else "gbm_tuning.json")
    policies, _, _ = data.load_dataset()
    train = policies[~policies["is_holdout"]]
    rng = np.random.default_rng(SEED)
    trials = []
    for i in range(N_TRIALS):
        params = sample_params(rng)
        start = time.time()
        deviance, n_rounds = gbm.cv_score(train, params, additive=additive, monotone=monotone)
        trials.append({"trial": i, "params": params, "cv_deviance": deviance, "n_rounds": n_rounds, "seconds": round(time.time() - start, 1)})
        print(trials[-1], flush=True)
        best = min(trials, key=lambda t: t["cv_deviance"])
        out_path.write_text(json.dumps({"variant": variant or "full", "n_trials_done": len(trials), "best": best, "trials": trials}, indent=2), encoding="utf-8")
    print("best:", best)


if __name__ == "__main__":
    main()
