"""Fit the final models (the eight rating factors other than BonusMalus) and score them on the holdout.

The specification was recorded in docs/DECISIONS.md (D31) and committed before this script was
first run. Nothing here is chosen by looking at the holdout.

Run from the repo root, after run_glm.py, run_gbm.py and `python src/tune_gbm.py final`:
    python src/run_final.py

Reads  outputs/predictions.parquet and outputs/gbm_tuning_final.json.
Writes outputs/metrics_final.json         fitted models, rebalancing factors, holdout metrics
       outputs/final_glm.json             the final GLM's coefficients (read by build_rating_engine.py)
       outputs/glm_final_frequency_summary.txt
       plots/double_lift_final_glm_vs_gbm.png
       and adds freq_ and pp_ columns for glm_final and gbm_final to outputs/predictions.parquet.
"""

import json

import numpy as np
import pandas as pd

import gbm
import metrics
from check_bonus_malus_timing import exposure_class
from config import SEED
from design import BANDS, FINAL_FREQUENCY_TERMS, Design, add_model_columns
from glm import fit_poisson, predict
from run_gbm import PLOTS, add_model, plot_double_lift
from run_glm import OUTPUTS, PREDICTIONS_PATH, fit_severity_model, holdout_metrics, rebalance_factor, summary_text

FINAL_MODELS = ["glm_final", "gbm_final"]


def coefficient_table(design, result):
    """Every level of every term with its coefficient; reference levels are listed with 0."""
    params = pd.Series(np.asarray(result.params), index=design.columns)
    rows = [{"term": "Intercept", "level": "", "coef": float(params["Intercept"])}]
    for term in design.terms:
        if term in design.levels:
            for level in design.levels[term]:
                rows.append({"term": term, "level": str(level), "coef": float(params.get(f"{term}[{level}]", 0.0))})
        else:
            rows.append({"term": term, "level": "", "coef": float(params[term])})
    return rows


def main():
    saved = pd.read_parquet(PREDICTIONS_PATH)
    saved = saved.drop(columns=[c for c in saved.columns if c.endswith("_final")])
    policies = add_model_columns(saved)
    train = policies[~policies["is_holdout"]]
    null_rate = float(train["ClaimNb"].sum() / train["Exposure"].sum())
    tuning = json.loads((OUTPUTS / "gbm_tuning_final.json").read_text())
    if tuning["features"] != gbm.FINAL_FEATURES or tuning["n_trials_done"] != 30:
        raise SystemExit("gbm_tuning_final.json is incomplete or was tuned on other features")

    # Final GLM: statsmodels, unpenalised, the baseline terms without BonusMalusBand.
    design = Design(FINAL_FREQUENCY_TERMS).fit(train)
    freq = fit_poisson(design.transform(train), train["ClaimNb"], train["Exposure"])
    policies["freq_glm_final"] = predict(freq, design.transform(policies))
    (OUTPUTS / "glm_final_frequency_summary.txt").write_text(summary_text(freq, design.columns), encoding="utf-8")

    # The shared severity GLM (driver age only). Refitted here for its coefficients; it is the
    # same model as sev_glm, which run_glm.py wrote. Its response is defined as in data.load_dataset.
    policies["AvgSevCapped"] = np.where(policies["ClaimRecords"] > 0, policies["TotalLossCapped"] / policies["ClaimRecords"].clip(lower=1), np.nan)
    sev_pred, sev_design, sev = fit_severity_model(policies)
    if not np.allclose(sev_pred, policies["sev_glm"], rtol=1e-10):
        raise SystemExit("severity refit does not match sev_glm in predictions.parquet")

    t = policies[~policies["is_holdout"]]
    glm_freq_factor = rebalance_factor(t["ClaimNb"], t["freq_glm_final"], t["Exposure"])
    glm_pp_factor = rebalance_factor(t["TotalLossCapped"], t["freq_glm_final"] * t["sev_glm"], t["Exposure"])
    policies["pp_glm_final"] = policies["freq_glm_final"] * policies["sev_glm"] * glm_pp_factor
    large_loss_load = float(t["TotalLoss"].sum() / t["TotalLossCapped"].sum())

    # Final GBM: the eight factors raw, tuned parameters from training cross-validation only.
    gbm_info = add_model(policies, "gbm_final", tuning, features=gbm.FINAL_FEATURES)

    result = {
        "glm_final": {"terms": FINAL_FREQUENCY_TERMS, "frequency_parameters": len(design.columns),
                      "frequency_rebalance_factor": glm_freq_factor, "pure_premium_rebalance_factor": glm_pp_factor,
                      "train_frequency_deviance": metrics.mean_poisson_deviance(t["ClaimNb"], t["freq_glm_final"], t["Exposure"])},
        "gbm_final": {"features": gbm.FINAL_FEATURES, **gbm_info},
        "large_loss_load": large_loss_load,
    }
    h = policies[policies["is_holdout"]]
    for name in FINAL_MODELS:
        result[name]["holdout"] = holdout_metrics(policies, f"freq_{name}", null_rate, f"pp_{name}")
        result[name]["holdout"]["frequency_a_over_e_by_exposure_class"] = metrics.actual_vs_expected(
            h["ClaimNb"], h[f"freq_{name}"], h["Exposure"], by=exposure_class(h["Exposure"]))["a_over_e"].round(4).to_dict()
    est, low, high = metrics.bootstrap_deviance_difference(h["ClaimNb"], h["freq_glm_final"], h["freq_gbm_final"], h["Exposure"], n_boot=1000, seed=SEED)
    result["gbm_minus_glm_deviance_reduction"] = {"estimate": est, "bootstrap_95_low": low, "bootstrap_95_high": high}
    lift = metrics.double_lift(h["freq_glm_final"], h["freq_gbm_final"], h["ClaimNb"], h["Exposure"], n_bins=10)
    plot_double_lift(lift, "GLM", "GBM", PLOTS / "double_lift_final_glm_vs_gbm.png")

    final_glm = {
        "description": "Final frequency and severity GLMs (statsmodels). Read by src/build_rating_engine.py.",
        "frequency": {"coefficients": coefficient_table(design, freq), "columns": design.columns},
        "severity": {"coefficients": coefficient_table(sev_design, sev), "columns": sev_design.columns},
        "bands": {name: BANDS[name][1] for name in BANDS if name in FINAL_FREQUENCY_TERMS + sev_design.terms},
        "band_sources": {name: BANDS[name][0] for name in BANDS if name in FINAL_FREQUENCY_TERMS + sev_design.terms},
        "frequency_rebalance_factor": glm_freq_factor,
        "pure_premium_rebalance_factor": glm_pp_factor,
        "large_loss_load": large_loss_load,
    }
    (OUTPUTS / "final_glm.json").write_text(json.dumps(final_glm, indent=2), encoding="utf-8")
    (OUTPUTS / "metrics_final.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    keep = [c for c in policies.columns if c in saved.columns or c.endswith(tuple(f"_{m}" for m in FINAL_MODELS))]
    policies[keep].to_parquet(PREDICTIONS_PATH, index=False)

    for name in FINAL_MODELS:
        r = result[name]
        print(name, {k: v for k, v in r.items() if k not in ("holdout", "terms", "features", "params")})
        print("  holdout frequency:", r["holdout"]["frequency"])
        print("  holdout pure premium:", r["holdout"]["pure_premium"])
        print("  A/E by exposure class:", r["holdout"]["frequency_a_over_e_by_exposure_class"])
    print("GBM minus GLM:", result["gbm_minus_glm_deviance_reduction"])
    print(lift.round(4).to_string())


if __name__ == "__main__":
    main()
