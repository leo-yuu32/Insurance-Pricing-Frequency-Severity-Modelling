"""Evidence for the monotone-constraint decision, using cross-validation on TRAINING data only.

Scores the tuned GBM parameters with no constraint on BonusMalus and with each of LightGBM's
three ways of enforcing an increasing constraint, and prints the results as a table.

Run from the repo root:  python src/compare_gbm_constraints.py
"""

import json

import pandas as pd

import data
import gbm
from run_glm import OUTPUTS


def main():
    policies, _, _ = data.load_dataset()
    train = policies[~policies["is_holdout"]]
    params = json.loads((OUTPUTS / "gbm_tuning_monotone.json").read_text())["best"]["params"]
    rows = []
    for label, monotone, extra in [
        ("no constraint", False, {}),
        ("increasing, method basic (LightGBM default)", True, {"monotone_constraints_method": "basic"}),
        ("increasing, method intermediate", True, {"monotone_constraints_method": "intermediate"}),
        ("increasing, method advanced", True, {"monotone_constraints_method": "advanced"}),
    ]:
        deviance, n_rounds = gbm.cv_score(train, {**params, **extra}, monotone=monotone)
        rows.append({"BonusMalus constraint": label, "cv_deviance": deviance, "n_rounds": n_rounds})
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    table["vs_no_constraint"] = table["cv_deviance"] - table["cv_deviance"].iloc[0]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
