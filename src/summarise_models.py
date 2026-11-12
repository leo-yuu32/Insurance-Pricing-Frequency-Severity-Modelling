"""One comparison table for every model, computed from the saved predictions.

Run from the repo root, after the model scripts:
    python src/summarise_models.py

Reads  outputs/predictions.parquet.
Writes outputs/model_comparison.json: holdout frequency metrics for every model, paired bootstrap
intervals for the differences that matter, and the share of the GLM-to-GBM gap each extended
GLM closes. No model is fitted here.
"""

import json

import pandas as pd

import metrics
from config import SEED
from run_glm import OUTPUTS, PREDICTIONS_PATH

MODELS = {
    "glm": "GLM (statsmodels)",
    "glm_int": "GLM + interaction",
    "glm_ladder": "GLM + ladder indicator",
    "glm_both": "GLM + ladder indicator + interaction",
    "gbm_additive": "Additive GBM (no interactions)",
    "gbm_monotone": "GBM, BonusMalus constrained",
    "gbm": "GBM",
}


def main():
    policies = pd.read_parquet(PREDICTIONS_PATH)
    train, h = policies[~policies["is_holdout"]], policies[policies["is_holdout"]]
    null_rate = float(train["ClaimNb"].sum() / train["Exposure"].sum())
    present = [m for m in MODELS if f"freq_{m}" in policies]

    def boot(a, b):
        est, low, high = metrics.bootstrap_deviance_difference(h["ClaimNb"], h[f"freq_{a}"], h[f"freq_{b}"], h["Exposure"], n_boot=1000, seed=SEED)
        return {"deviance_reduction": est, "bootstrap_95_low": low, "bootstrap_95_high": high}

    table = {}
    for m in present:
        row = metrics.frequency_summary(h["ClaimNb"], h[f"freq_{m}"], h["Exposure"], null_rate)
        row["label"] = MODELS[m]
        row["train_deviance"] = metrics.mean_poisson_deviance(train["ClaimNb"], train[f"freq_{m}"], train["Exposure"])
        row["pure_premium_gini_capped_losses"] = metrics.gini(h[f"pp_{m}"], h["TotalLossCapped"], h["Exposure"])
        row["pure_premium_a_over_e_capped_losses"] = float(h["TotalLossCapped"].sum() / (h[f"pp_{m}"] * h["Exposure"]).sum())
        table[m] = row
    dev = {m: table[m]["poisson_deviance"] for m in present}
    gini = {m: table[m]["gini"] for m in present}

    result = {"holdout_policies": int(len(h)), "null_deviance": table["glm"]["null_deviance"], "models": table, "differences": {}, "gap_closed": {}}
    pairs = [("glm", m) for m in present if m != "glm"] + [(m, "gbm") for m in ["glm_int", "glm_ladder", "glm_both", "gbm_additive"] if m in present]
    for a, b in pairs:
        result["differences"][f"{a}_to_{b}"] = boot(a, b)

    if "gbm" in present:
        gap = dev["glm"] - dev["gbm"]
        result["gap_closed"]["glm_to_gbm_gap_deviance"] = gap
        for m in ["glm_int", "glm_ladder", "glm_both"]:
            if m in present:
                result["gap_closed"][m] = {
                    "share_of_gap_closed_deviance": (dev["glm"] - dev[m]) / gap,
                    "share_of_gap_closed_gini": (gini[m] - gini["glm"]) / (gini["gbm"] - gini["glm"]),
                }
        if "gbm_additive" in present:
            shape, inter = dev["glm"] - dev["gbm_additive"], dev["gbm_additive"] - dev["gbm"]
            result["gap_closed"]["decomposition"] = {
                "shape_portion_glm_to_additive_gbm": shape, "interaction_portion_additive_gbm_to_gbm": inter,
                "shape_share_of_gap": shape / gap, "interaction_share_of_gap": inter / gap,
            }
            if "glm_int" in present:
                result["gap_closed"]["glm_int"]["share_of_interaction_portion_closed"] = (dev["glm"] - dev["glm_int"]) / inter
            if "glm_ladder" in present:
                result["gap_closed"]["glm_ladder"]["share_of_shape_portion_closed"] = (dev["glm"] - dev["glm_ladder"]) / shape
            if "glm_both" in present and "glm_ladder" in present:
                result["gap_closed"]["glm_both"]["share_of_interaction_portion_closed_by_its_interaction"] = (dev["glm_ladder"] - dev["glm_both"]) / inter

    (OUTPUTS / "model_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    cols = ["label", "poisson_deviance", "deviance_explained", "gini", "a_over_e", "train_deviance", "pure_premium_gini_capped_losses"]
    print(pd.DataFrame(table).T[cols].to_string(float_format=lambda v: f"{v:.5f}"))
    print(json.dumps(result["differences"], indent=1))
    print(json.dumps(result["gap_closed"], indent=1))


if __name__ == "__main__":
    main()
