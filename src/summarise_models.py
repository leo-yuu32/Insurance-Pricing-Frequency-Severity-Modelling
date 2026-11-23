"""One results file for every model, computed from the saved predictions. No model is fitted here.

Run from the repo root, after the model scripts:
    python src/summarise_models.py

Reads  outputs/predictions.parquet.
Writes outputs/model_comparison.json, which src/build_dashboard_data.py reads. It has three parts:
  final               the final GLM and GBM (eight factors, without BonusMalus): holdout metrics,
                      the GLM-minus-GBM deviance with a bootstrap interval, the double-lift table
  with_bonus_malus    the nine-factor models, kept as a labelled comparison
  other_models_scored_on_the_holdout
                      the remaining models that were scored on the holdout, so that every use
                      of the holdout is on record
"""

import json

import pandas as pd

import metrics
from config import SEED
from run_glm import OUTPUTS, PREDICTIONS_PATH

FINAL = {"glm_final": "GLM", "gbm_final": "GBM"}
WITH_BONUS_MALUS = {
    "glm": "GLM",
    "glm_ladder": "GLM + ladder indicator",
    "gbm_additive": "Additive GBM (no interactions)",
    "gbm": "GBM",
}
OTHERS = {
    "glm_int": "GLM + BonusMalus x Region interaction",
    "glm_both": "GLM + ladder indicator + interaction",
    "gbm_monotone": "GBM with BonusMalus constrained to be increasing",
}


def model_row(train, holdout, name, label):
    """Holdout metrics for one model: frequency first, pure premium second."""
    h, e = holdout, holdout["Exposure"]
    null_rate = float(train["ClaimNb"].sum() / train["Exposure"].sum())
    row = {"label": label, **metrics.frequency_summary(h["ClaimNb"], h[f"freq_{name}"], e, null_rate)}
    row["train_deviance"] = metrics.mean_poisson_deviance(train["ClaimNb"], train[f"freq_{name}"], train["Exposure"])
    pp = h[f"pp_{name}"]
    without_largest = h["TotalLoss"] < h["TotalLoss"].max()
    row["pure_premium_a_over_e_capped_losses"] = float(h["TotalLossCapped"].sum() / (pp * e).sum())
    row["pure_premium_gini_capped_losses"] = metrics.gini(pp, h["TotalLossCapped"], e)
    row["pure_premium_gini_all_losses"] = metrics.gini(pp, h["TotalLoss"], e)
    row["pure_premium_gini_without_largest_policy"] = metrics.gini(pp[without_largest], h["TotalLoss"][without_largest], e[without_largest])
    return row


def difference(holdout, a, b):
    """Deviance of A minus deviance of B on the holdout, with a paired bootstrap interval (positive: B is better)."""
    est, low, high = metrics.bootstrap_deviance_difference(
        holdout["ClaimNb"], holdout[f"freq_{a}"], holdout[f"freq_{b}"], holdout["Exposure"], n_boot=1000, seed=SEED)
    return {"deviance_reduction": est, "bootstrap_95_low": low, "bootstrap_95_high": high}


def build_summary(policies):
    train, h = policies[~policies["is_holdout"]], policies[policies["is_holdout"]]
    present = {m for m in {**FINAL, **WITH_BONUS_MALUS, **OTHERS} if f"freq_{m}" in policies}
    summary = {
        "holdout": {"policies": int(len(h)), "exposure": float(h["Exposure"].sum()), "claims": int(h["ClaimNb"].sum())},
        "train": {"policies": int(len(train)), "exposure": float(train["Exposure"].sum()), "claims": int(train["ClaimNb"].sum())},
    }

    if set(FINAL) <= present:
        lift = metrics.double_lift(h["freq_glm_final"], h["freq_gbm_final"], h["ClaimNb"], h["Exposure"], n_bins=10)
        lift = lift.rename(columns={"rate_a": "rate_glm", "rate_b": "rate_gbm"}).reset_index()
        summary["final"] = {
            "title": "Final models: eight rating factors, without BonusMalus",
            "models": {m: model_row(train, h, m, label) for m, label in FINAL.items()},
            "glm_minus_gbm": difference(h, "glm_final", "gbm_final"),
            "double_lift": lift.to_dict(orient="records"),
        }

    nine = [m for m in WITH_BONUS_MALUS if m in present]
    if "glm" in nine and "gbm" in nine:
        dev = {}
        block = {"title": "Comparison with BonusMalus (nine factors): set aside",
                 "models": {}, "differences": {}, "gap_closed": {}}
        for m in nine:
            block["models"][m] = model_row(train, h, m, WITH_BONUS_MALUS[m])
            dev[m] = block["models"][m]["poisson_deviance"]
        pairs = [("glm", "gbm")] + [(a, b) for a, b in [("glm", "glm_ladder"), ("glm_ladder", "gbm"), ("glm", "gbm_additive"), ("gbm_additive", "gbm")]
                                    if a in nine and b in nine]
        for a, b in pairs:
            block["differences"][f"{a}_to_{b}"] = difference(h, a, b)
        gap = dev["glm"] - dev["gbm"]
        block["gap_closed"]["glm_to_gbm_gap_deviance"] = gap
        if "glm_ladder" in nine:
            g = {m: block["models"][m]["gini"] for m in ("glm", "glm_ladder", "gbm")}
            block["gap_closed"]["glm_ladder"] = {
                "share_of_gap_closed_deviance": (dev["glm"] - dev["glm_ladder"]) / gap,
                "share_of_gap_closed_gini": (g["glm_ladder"] - g["glm"]) / (g["gbm"] - g["glm"]),
            }
        if "gbm_additive" in nine:
            shape, inter = dev["glm"] - dev["gbm_additive"], dev["gbm_additive"] - dev["gbm"]
            block["gap_closed"]["decomposition"] = {"shape_portion": shape, "interaction_portion": inter,
                                                    "shape_share_of_gap": shape / gap, "interaction_share_of_gap": inter / gap}
        summary["with_bonus_malus"] = block
        if set(FINAL) <= present:
            # Positive: the nine-factor model scores better than the final model of the same kind.
            summary["cost_of_leaving_out_bonus_malus"] = {"glm": difference(h, "glm_final", "glm"), "gbm": difference(h, "gbm_final", "gbm")}

    summary["other_models_scored_on_the_holdout"] = {
        m: {"label": label, "poisson_deviance": metrics.mean_poisson_deviance(h["ClaimNb"], h[f"freq_{m}"], h["Exposure"]),
            "gini": metrics.gini(h[f"freq_{m}"], h["ClaimNb"], h["Exposure"])}
        for m, label in OTHERS.items() if m in present
    }
    return summary


def main():
    summary = build_summary(pd.read_parquet(PREDICTIONS_PATH))
    (OUTPUTS / "model_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cols = ["label", "poisson_deviance", "deviance_explained", "gini", "a_over_e", "train_deviance",
            "pure_premium_gini_capped_losses", "pure_premium_gini_without_largest_policy"]
    for part in ["final", "with_bonus_malus"]:
        if part in summary:
            print(summary[part]["title"])
            print(pd.DataFrame(summary[part]["models"]).T[cols].to_string(float_format=lambda v: f"{v:.5f}"))
    print(json.dumps({k: summary.get(k) for k in ["cost_of_leaving_out_bonus_malus", "other_models_scored_on_the_holdout"]}, indent=1))
    if "final" in summary:
        print("GLM minus GBM:", summary["final"]["glm_minus_gbm"])
        print(pd.DataFrame(summary["final"]["double_lift"]).round(4).to_string(index=False))
    if "with_bonus_malus" in summary:
        print(json.dumps(summary["with_bonus_malus"]["differences"], indent=1))
        print(json.dumps(summary["with_bonus_malus"]["gap_closed"], indent=1))


if __name__ == "__main__":
    main()
