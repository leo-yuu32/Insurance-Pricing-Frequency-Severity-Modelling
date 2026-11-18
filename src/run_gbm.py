"""Fit the tuned LightGBM frequency model and compare it with the GLM on the holdout.

Run from the repo root, after run_glm.py and tune_gbm.py:
    python src/run_gbm.py

Reads  outputs/predictions.parquet and outputs/gbm_tuning.json (and the _monotone and _additive
       tuning files if present).
Writes outputs/metrics_gbm.json, plots/double_lift_glm_vs_gbm.png, and adds the GBM columns
to outputs/predictions.parquet.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import gbm  # noqa: E402
import metrics  # noqa: E402
from design import add_model_columns  # noqa: E402
from run_glm import OUTPUTS, PREDICTIONS_PATH, holdout_metrics, rebalance_factor  # noqa: E402

PLOTS = OUTPUTS.parent / "plots"


def add_model(policies, name, tuning, additive=False, monotone=False, features=None):
    """Fit one GBM on the training rows and add freq_<name> and pp_<name>. Returns its factors."""
    train = policies[~policies["is_holdout"]]
    best = tuning["best"]
    model = gbm.fit(train, best["params"], best["n_rounds"], additive=additive, monotone=monotone, features=features)
    raw = gbm.predict_rate(model, policies)
    freq_factor = rebalance_factor(train["ClaimNb"], raw[~policies["is_holdout"].to_numpy()], train["Exposure"])
    policies[f"freq_{name}"] = raw * freq_factor
    train = policies[~policies["is_holdout"]]
    pp_factor = rebalance_factor(train["TotalLossCapped"], train[f"freq_{name}"] * train["sev_glm"], train["Exposure"])
    policies[f"pp_{name}"] = policies[f"freq_{name}"] * policies["sev_glm"] * pp_factor
    return {"params": best["params"], "n_rounds": best["n_rounds"], "cv_deviance": best["cv_deviance"],
            "frequency_rebalance_factor": freq_factor, "pure_premium_rebalance_factor": pp_factor,
            "train_frequency_deviance": metrics.mean_poisson_deviance(train["ClaimNb"], train[f"freq_{name}"], train["Exposure"])}


def plot_double_lift(table, label_a, label_b, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = table.index
    ax.plot(x, table["actual_rate"], color="black", marker="o", linewidth=2, label="Actual")
    ax.plot(x, table["rate_a"], color="#1f77b4", marker="s", linewidth=2, label=label_a)
    ax.plot(x, table["rate_b"], color="#d62728", marker="^", linewidth=2, label=label_b)
    ax.set_xticks(list(x))
    ax.set_xlabel(f"Decile of {label_b} / {label_a} prediction (equal exposure; 1 = {label_b} lowest relative to {label_a})")
    ax.set_ylabel("Claims per policy-year (holdout)")
    ax.set_title(f"Double-lift chart: {label_a} versus {label_b}, claim frequency")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    saved = pd.read_parquet(PREDICTIONS_PATH)
    saved = saved.drop(columns=[c for c in saved.columns if c.endswith(("_gbm", "_gbm_additive", "_gbm_monotone", "_gbm_unconstrained"))])
    # (the final models' columns end in "_final" and are left alone; run_final.py writes them)
    policies = add_model_columns(saved)
    train = policies[~policies["is_holdout"]]
    null_rate = float(train["ClaimNb"].sum() / train["Exposure"].sum())
    result = {}

    tuning = json.loads((OUTPUTS / "gbm_tuning.json").read_text())
    result["gbm"] = add_model(policies, "gbm", tuning)
    models = ["gbm"]
    # The same model with BonusMalus constrained to be increasing, tuned separately, to record
    # what the constraint costs (docs/DECISIONS.md).
    monotone_path = OUTPUTS / "gbm_tuning_monotone.json"
    if monotone_path.exists():
        result["gbm_monotone"] = add_model(policies, "gbm_monotone", json.loads(monotone_path.read_text()), monotone=True)
        models.append("gbm_monotone")
    additive_path = OUTPUTS / "gbm_tuning_additive.json"
    if additive_path.exists():
        additive_tuning = json.loads(additive_path.read_text())
        if additive_tuning["n_trials_done"] == tuning["n_trials_done"]:  # only once its search is complete
            result["gbm_additive"] = add_model(policies, "gbm_additive", additive_tuning, additive=True)
            models.append("gbm_additive")
    for name in models:
        result[name]["holdout"] = holdout_metrics(policies, f"freq_{name}", null_rate, f"pp_{name}")

    h = policies[policies["is_holdout"]]
    glm_dev = metrics.mean_poisson_deviance(h["ClaimNb"], h["freq_glm"], h["Exposure"])
    result["comparison_with_glm"] = {}
    for name in models:
        est, low, high = metrics.bootstrap_deviance_difference(h["ClaimNb"], h["freq_glm"], h[f"freq_{name}"], h["Exposure"], n_boot=1000, seed=gbm.SEED)
        result["comparison_with_glm"][name] = {"glm_deviance": glm_dev, "deviance_reduction": est, "bootstrap_95_low": low, "bootstrap_95_high": high}

    lift = metrics.double_lift(h["freq_glm"], h["freq_gbm"], h["ClaimNb"], h["Exposure"], n_bins=10)
    result["double_lift_glm_vs_gbm"] = lift.round(6).reset_index().to_dict(orient="records")
    plot_double_lift(lift, "GLM", "GBM", PLOTS / "double_lift_glm_vs_gbm.png")

    (OUTPUTS / "metrics_gbm.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    keep = [c for c in policies.columns if c in saved.columns or c.startswith(("freq_", "pp_"))]
    policies[keep].to_parquet(PREDICTIONS_PATH, index=False)

    for name in models:
        r = result[name]
        print(name, {k: r[k] for k in ["n_rounds", "cv_deviance", "frequency_rebalance_factor", "pure_premium_rebalance_factor", "train_frequency_deviance"]})
        print("  holdout frequency:", r["holdout"]["frequency"])
        print("  holdout pure premium:", r["holdout"]["pure_premium"])
        print("  vs GLM:", result["comparison_with_glm"][name])
    print(lift.round(4).to_string())


if __name__ == "__main__":
    main()
