"""Fit the baseline GLMs and write their predictions and holdout metrics.

Run from the repo root:  python src/run_glm.py

Writes:
  outputs/predictions.parquet        one row per policy, predictions from every model (git-ignored)
  outputs/metrics_glm.json           holdout metrics for the baseline GLM
  outputs/glm_frequency_summary.txt  statsmodels coefficient table, frequency
  outputs/glm_severity_summary.txt   statsmodels coefficient table, severity
  artifacts/holdout_idpol.txt        the holdout policy IDs
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
import metrics
from design import FREQUENCY_TERMS, SEVERITY_TERMS, Design, add_model_columns
from glm import fit_gamma, fit_poisson, predict

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
PREDICTIONS_PATH = OUTPUTS / "predictions.parquet"
AE_FACTORS = ["DrivAgeBand", "BonusMalusBand", "VehAgeBand", "VehPowerBand", "Area", "VehBrand", "VehGas"]
KEEP = ["IDpol", "fold", "is_holdout", "Exposure", "ClaimNb", "ClaimRecords", "TotalLoss", "TotalLossCapped"] + data.RATING_FACTORS


def rebalance_factor(actual, pred_rate, exposure):
    """Training actual over training expected. Applied as one multiplier to the whole model."""
    return float(np.sum(actual) / np.sum(np.asarray(pred_rate) * np.asarray(exposure)))


def fit_severity_model(policies):
    """Shared by every frequency model. Returns (predicted severity per policy, design, result)."""
    train_claims = policies[(~policies["is_holdout"]) & (policies["ClaimRecords"] > 0)]
    design = Design(SEVERITY_TERMS).fit(train_claims)
    result = fit_gamma(design.transform(train_claims), train_claims["AvgSevCapped"], train_claims["ClaimRecords"])
    return predict(result, design.transform(policies)), design, result


def holdout_metrics(policies, freq_col, null_rate, pure_premium_col=None):
    """Primary frequency metrics on the holdout, plus secondary pure premium metrics if given."""
    h = policies[policies["is_holdout"]]
    out = {"frequency": metrics.frequency_summary(h["ClaimNb"], h[freq_col], h["Exposure"], null_rate)}
    out["frequency_a_over_e_by_factor"] = {
        f: metrics.actual_vs_expected(h["ClaimNb"], h[freq_col], h["Exposure"], by=h[f])["a_over_e"].round(4).to_dict() for f in AE_FACTORS
    }
    if pure_premium_col is not None:
        pp, e = h[pure_premium_col], h["Exposure"]
        no_top = h["TotalLoss"] < h["TotalLoss"].max()
        t = policies[~policies["is_holdout"]]
        load = float(t["TotalLoss"].sum() / t["TotalLossCapped"].sum())
        out["pure_premium"] = {
            "a_over_e_capped_losses": float(h["TotalLossCapped"].sum() / (pp * e).sum()),
            "a_over_e_all_losses_after_large_loss_load": float(h["TotalLoss"].sum() / (pp * e * load).sum()),
            "gini_capped_losses": metrics.gini(pp, h["TotalLossCapped"], e),
            "gini_all_losses": metrics.gini(pp, h["TotalLoss"], e),
            "gini_excluding_largest_policy": metrics.gini(pp[no_top], h["TotalLoss"][no_top], e[no_top]),
        }
    return out


def summary_text(result, columns):
    table = pd.DataFrame({"coef": result.params, "std_err": result.bse, "z": result.tvalues, "p": result.pvalues,
                          "relativity": np.exp(result.params)}, index=columns)
    head = f"n = {int(result.nobs)}, deviance = {result.deviance:.2f}, Pearson chi2 / df = {result.pearson_chi2 / result.df_resid:.4f}\n\n"
    return head + table.to_string(float_format=lambda v: f"{v:.5f}") + "\n"


def main():
    OUTPUTS.mkdir(exist_ok=True)
    policies, claims, cap = data.load_dataset()
    data.save_holdout_ids(policies)
    policies = add_model_columns(policies)
    train = policies[~policies["is_holdout"]]

    # Frequency
    design = Design(FREQUENCY_TERMS).fit(train)
    freq = fit_poisson(design.transform(train), train["ClaimNb"], train["Exposure"])
    policies["freq_glm"] = predict(freq, design.transform(policies))
    (OUTPUTS / "glm_frequency_summary.txt").write_text(summary_text(freq, design.columns), encoding="utf-8")

    # Severity (shared by every frequency model)
    policies["sev_glm"], sev_design, sev = fit_severity_model(policies)
    (OUTPUTS / "glm_severity_summary.txt").write_text(summary_text(sev, sev_design.columns), encoding="utf-8")

    # Pure premium on the capped basis, rebalanced to capped training losses. The losses above
    # the cap are reported separately as one flat large-loss load (see docs/DECISIONS.md).
    train = policies[~policies["is_holdout"]]
    freq_factor = rebalance_factor(train["ClaimNb"], train["freq_glm"], train["Exposure"])
    pp_factor = rebalance_factor(train["TotalLossCapped"], train["freq_glm"] * train["sev_glm"], train["Exposure"])
    policies["pp_glm"] = policies["freq_glm"] * policies["sev_glm"] * pp_factor

    null_rate = float(train["ClaimNb"].sum() / train["Exposure"].sum())
    holdout = policies[policies["is_holdout"]]
    result = {
        "data": {
            "policies": int(len(policies)), "train_policies": int(len(train)), "holdout_policies": int(len(holdout)),
            "train_exposure": float(train["Exposure"].sum()), "holdout_exposure": float(holdout["Exposure"].sum()),
            "train_claims": int(train["ClaimNb"].sum()), "holdout_claims": int(holdout["ClaimNb"].sum()),
            "train_frequency": null_rate, "holdout_frequency": float(holdout["ClaimNb"].sum() / holdout["Exposure"].sum()),
            "large_loss_cap": cap, "train_claims_capped": int((claims.loc[~claims["is_holdout"], "ClaimAmount"] > cap).sum()),
            "train_loss": float(train["TotalLoss"].sum()), "holdout_loss": float(holdout["TotalLoss"].sum()),
            "train_loss_capped": float(train["TotalLossCapped"].sum()), "holdout_loss_capped": float(holdout["TotalLossCapped"].sum()),
            "large_loss_load": float(train["TotalLoss"].sum() / train["TotalLossCapped"].sum()),
            "largest_train_claim_share_of_train_loss": float(claims.loc[~claims["is_holdout"], "ClaimAmount"].max() / train["TotalLoss"].sum()),
        },
        "glm": {
            "frequency_parameters": int(len(design.columns)), "severity_parameters": int(len(sev_design.columns)),
            "frequency_rebalance_factor": freq_factor, "pure_premium_rebalance_factor": pp_factor,
            "severity_dispersion_pearson": float(sev.pearson_chi2 / sev.df_resid),
            "train_frequency_deviance": metrics.mean_poisson_deviance(train["ClaimNb"], train["freq_glm"], train["Exposure"]),
            "holdout": holdout_metrics(policies, "freq_glm", null_rate, "pp_glm"),
        },
    }
    (OUTPUTS / "metrics_glm.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    policies[KEEP + ["freq_glm", "sev_glm", "pp_glm"]].to_parquet(PREDICTIONS_PATH, index=False)

    print(json.dumps({k: v for k, v in result["glm"].items() if k != "holdout"}, indent=2))
    print(json.dumps(result["glm"]["holdout"]["frequency"], indent=2))
    print(json.dumps(result["glm"]["holdout"]["pure_premium"], indent=2))
    for f in ["DrivAgeBand", "BonusMalusBand"]:
        print(f, result["glm"]["holdout"]["frequency_a_over_e_by_factor"][f])


if __name__ == "__main__":
    main()
