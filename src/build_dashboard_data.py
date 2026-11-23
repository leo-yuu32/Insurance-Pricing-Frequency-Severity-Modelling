"""Precompute everything the dashboard shows, as small JSON files in artifacts/dashboard/.

Run from the repo root, after the model scripts and summarise_models.py (about a minute):
    python src/build_dashboard_data.py

Reads  outputs/predictions.parquet and the saved results in outputs/ (model_comparison.json,
       metrics_glm.json, metrics_gbm.json, metrics_interactions.json, metrics_final.json,
       final_glm.json and bonus_malus_timing.json).
Writes artifacts/dashboard/*.json, the only files the dashboard reads.

No model is fitted here and nothing is chosen. Every number is either copied from a saved result
or is a descriptive summary of the saved predictions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import metrics
from check_bonus_malus_timing import exposure_class
from run_glm import OUTPUTS, PREDICTIONS_PATH

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = ROOT / "artifacts" / "dashboard"

FACTOR_NAMES = {"BonusMalus": "No-claims score", "Density": "Population density", "DrivAge": "Driver age",
                "Region": "Region", "VehGas": "Fuel type", "VehAge": "Vehicle age", "VehBrand": "Vehicle brand",
                "VehPower": "Vehicle power", "Area": "Area type"}
TERM_NAMES = {"DrivAgeBand": "Driver age", "VehAgeBand": "Vehicle age", "VehPowerBand": "Vehicle power",
              "LogDensity": "Population density", "VehBrand": "Vehicle brand", "VehGas": "Fuel type", "Region": "Region"}
BANDED_TERMS = {"DrivAgeBand", "VehAgeBand", "VehPowerBand"}
AGE_RANGE = (18, 85)              # older ages have fewer than 61 policy-years each in the holdout
SCORE_RANGE = (50, 100)
MIN_POLICIES_PER_SCORE = 300      # full-year training records needed to show a score
PRICE_RATIO_EDGES = np.round(np.arange(0.60, 1.5001, 0.02), 2)
PRICE_THRESHOLD = 0.10
LORENZ_POINTS = 201
LIFT_GROUPS = 10
STAIRCASE_VALUES = {"50": "50", "62 (surcharged at the last renewal)": "62", "58 (one renewal later)": "58",
                    "55 (two renewals later)": "55", "52 (three renewals later)": "52"}


def relative_reduction(reduction, reference_deviance):
    """A deviance reduction (reference model minus this model) and its 95% interval, as percentages
    of the reference model's holdout deviance. Positive means this model's prediction error is lower."""
    return {"pct": 100.0 * reduction["deviance_reduction"] / reference_deviance,
            "low_pct": 100.0 * reduction["bootstrap_95_low"] / reference_deviance,
            "high_pct": 100.0 * reduction["bootstrap_95_high"] / reference_deviance}


def _interval(d):
    """The saved intervals use two key layouts; return the one used by relative_reduction."""
    est = d["deviance_reduction"] if "deviance_reduction" in d else d["estimate"]
    return {"deviance_reduction": est, "bootstrap_95_low": d["bootstrap_95_low"], "bootstrap_95_high": d["bootstrap_95_high"]}


def model_rows(saved):
    """One row per model scored on the holdout, copied from the saved results. Each reduction is
    measured against the GLM with the same rating factors."""
    final, gbm9, inter = saved["metrics_final"], saved["metrics_gbm"], saved["metrics_interactions"]
    specs = [
        ("glm_final", "final", "GLM", "GLM", final["glm_final"], None),
        ("gbm_final", "final", "GBM", "GBM", final["gbm_final"], final["gbm_minus_glm_deviance_reduction"]),
        ("glm", "with_score", "GLM", "GLM", saved["metrics_glm"]["glm"], None),
        ("glm_int", "with_score", "GLM + score by region", "GLM",
         inter["models"]["glm_int"], inter["gap"]["glm_int"]["glm_to_model"]),
        ("glm_ladder", "with_score", "GLM + ladder indicator", "GLM",
         inter["models"]["glm_ladder"], inter["gap"]["glm_ladder"]["glm_to_model"]),
        ("glm_both", "with_score", "GLM + ladder + score by region", "GLM",
         inter["models"]["glm_both"], inter["gap"]["glm_both"]["glm_to_model"]),
        ("gbm_additive", "with_score", "Additive GBM", "GBM",
         gbm9["gbm_additive"], gbm9["comparison_with_glm"]["gbm_additive"]),
        ("gbm_monotone", "with_score", "Monotone GBM", "GBM",
         gbm9["gbm_monotone"], gbm9["comparison_with_glm"]["gbm_monotone"]),
        ("gbm", "with_score", "GBM", "GBM", gbm9["gbm"], gbm9["comparison_with_glm"]["gbm"]),
    ]
    reference = {"final": final["glm_final"]["holdout"]["frequency"]["poisson_deviance"],
                 "with_score": saved["metrics_glm"]["glm"]["holdout"]["frequency"]["poisson_deviance"]}
    rows = []
    for key, factor_set, label, kind, result, reduction in specs:
        f = result["holdout"]["frequency"]
        row = {"key": key, "factor_set": factor_set, "label": label, "kind": kind,
               "poisson_deviance": f["poisson_deviance"], "deviance_explained": f["deviance_explained"],
               "gini": f["gini"], "a_over_e": f["a_over_e"], "reference": reduction is None}
        if reduction is not None:
            r = _interval(reduction)
            row.update(reduction=r["deviance_reduction"], reduction_low=r["bootstrap_95_low"],
                       reduction_high=r["bootstrap_95_high"], **relative_reduction(r, reference[factor_set]))
        rows.append(row)
    return rows


def rates_by(df, key, predictions, lo, hi):
    """Exposure-weighted actual and predicted claim rates for each value of `key` from lo to hi.
    `predictions` maps an output name to a column of predicted rates."""
    d = df[(df[key] >= lo) & (df[key] <= hi)]
    g = d.groupby(key).agg(exposure=("Exposure", "sum"), claims=("ClaimNb", "sum"))
    g["actual"] = g["claims"] / g["exposure"]
    for name, col in predictions.items():
        g[name] = (d[col] * d["Exposure"]).groupby(d[key]).sum() / g["exposure"]
    return g.reset_index()


def claim_rate_by_score(df, lo, hi, min_policies):
    """Claims per policy-year by no-claims score, in full-year records, for the scores from lo to hi
    that have at least `min_policies` full-year records."""
    full = df[exposure_class(df["Exposure"]) == "full year"]
    g = full.groupby("BonusMalus").agg(policies=("ClaimNb", "size"), exposure=("Exposure", "sum"),
                                       claims=("ClaimNb", "sum"))
    g = g[(g.index >= lo) & (g.index <= hi) & (g["policies"] >= min_policies)]
    g["rate"] = g["claims"] / g["exposure"]
    g.index = g.index.astype(int)
    return g.rename_axis("score").reset_index()


def price_ratio_summary(ratio, exposure, edges, threshold):
    """Exposure-weighted histogram of one price over another, and the share of exposure where the
    two differ by more than `threshold` (ratio above 1 + threshold or below 1 - threshold).
    Ratios beyond the edges are counted in the end bins."""
    ratio = np.asarray(ratio, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    total = exposure.sum()
    clipped = np.clip(ratio, edges[0], edges[-1] - 1e-12)
    counts, _ = np.histogram(clipped, bins=edges, weights=exposure)
    bins = [{"low": float(lo), "high": float(hi), "share": float(c / total),
             "differs": bool(hi <= 1 - threshold + 1e-9 or lo >= 1 + threshold - 1e-9)}
            for lo, hi, c in zip(edges[:-1], edges[1:], counts)]
    higher, lower = ratio > 1 + threshold, ratio < 1 - threshold
    return {"bins": bins, "threshold": threshold,
            "share_differing": float(exposure[higher | lower].sum() / total),
            "share_higher": float(exposure[higher].sum() / total),
            "share_lower": float(exposure[lower].sum() / total),
            "share_beyond_axis": float(exposure[(ratio < edges[0]) | (ratio >= edges[-1])].sum() / total)}


def lift_by_average(pred_rate_a, pred_rate_b, actual, exposure, n_bins):
    """Lift table for two models together: policies sorted by the average of the two predicted rates
    and cut into n_bins groups of equal exposure (as in metrics.double_lift), from the lowest
    predicted risk to the highest. Each group has its actual rate and each model's average rate."""
    d = pd.DataFrame({"a": np.asarray(pred_rate_a, dtype=float), "b": np.asarray(pred_rate_b, dtype=float),
                      "actual": np.asarray(actual, dtype=float), "exposure": np.asarray(exposure, dtype=float)})
    d = d.assign(score=(d["a"] + d["b"]) / 2.0).sort_values("score", kind="mergesort")
    cum = (d["exposure"].cumsum() - d["exposure"] / 2.0) / d["exposure"].sum()
    d["bin"] = np.minimum((cum * n_bins).astype(int), n_bins - 1) + 1
    d["exp_a"], d["exp_b"] = d["a"] * d["exposure"], d["b"] * d["exposure"]
    g = d.groupby("bin").agg(exposure=("exposure", "sum"), actual=("actual", "sum"), exp_a=("exp_a", "sum"),
                             exp_b=("exp_b", "sum"))
    return pd.DataFrame({"bin": g.index, "exposure": g["exposure"].to_numpy(), "actual_rate": (g["actual"] / g["exposure"]).to_numpy(),
                         "rate_a": (g["exp_a"] / g["exposure"]).to_numpy(), "rate_b": (g["exp_b"] / g["exposure"]).to_numpy()})


def lorenz_points(pred_rate, actual, exposure, n_points):
    """The Lorenz curve of metrics.lorenz_curve, read off at n_points evenly spaced exposure shares."""
    x, y = metrics.lorenz_curve(pred_rate, actual, exposure)
    grid = np.linspace(0.0, 1.0, n_points)
    return grid, np.interp(grid, x, y)


def _band_order(level):
    return int("".join(ch for ch in level.split("-")[0] if ch.isdigit()))


def glm_relativities(coefficients, density_grid, density_reference):
    """Relativity exp(coef) of every level of every factor of the frequency GLM. The reference level
    (coefficient 0) has relativity 1. Banded factors keep their natural order and the others are
    sorted by relativity. Density is continuous on the log scale, and so it is given as a curve
    relative to density_reference."""
    factors, beta = {}, None
    for c in coefficients:
        if c["term"] == "Intercept":
            continue
        if c["term"] == "LogDensity":
            beta = c["coef"]
            continue
        factors.setdefault(c["term"], []).append(
            {"level": c["level"], "relativity": float(np.exp(c["coef"])), "reference": c["coef"] == 0.0})
    out = []
    for term, levels in factors.items():
        if term in BANDED_TERMS:
            levels = sorted(levels, key=lambda r: _band_order(r["level"]))
        elif len(levels) > 2:
            levels = sorted(levels, key=lambda r: -r["relativity"])
        out.append({"term": term, "name": TERM_NAMES[term], "levels": levels})
    curve = [{"density": float(d), "relativity": float((d / density_reference) ** beta)} for d in density_grid]
    density = {"term": "LogDensity", "name": TERM_NAMES["LogDensity"], "coefficient": beta,
               "reference_density": float(density_reference), "per_tenfold": float(10.0 ** beta), "curve": curve}
    return {"factors": out, "density": density}


def timing_checks(timing):
    """The three BonusMalus timing checks (training rows only), from the saved results."""
    by_class = {}
    for r in timing["by_exposure_class_below_100"]:
        by_class.setdefault(r["ExposureClass"], {})["off" if r["OffLadder"] else "on"] = r
    relativity = timing["glm_off_ladder_relativity_by_exposure_class"]
    check1 = [{"record_length": k, "rate_on_path": v["on"]["frequency"], "rate_off_path": v["off"]["frequency"],
               "ratio": v["off"]["frequency"] / v["on"]["frequency"],
               "glm_relativity": relativity[f"OffLadder[{k}]"], "claims_off_path": v["off"]["claims"]}
              for k, v in by_class.items()]
    check2 = [{"group": r["ValueGroup"], "policies": r["policies"], "claims": r["claims"],
               "share_with_claim": r["share_with_claim"]} for r in timing["full_year_records_by_value"]]
    check3 = [{"block": r["IDpolBlock"], "value": STAIRCASE_VALUES[r["ValueGroup"]], "claims": r["claims"],
               "rate": r["frequency"]} for r in timing["by_idpol_block_and_value"] if r["ValueGroup"] in STAIRCASE_VALUES]
    return {"rows": timing["rows"], "check1_record_length": check1, "check2_full_year_share": check2,
            "check3_staircase": check3}


def build(policies, saved):
    """Every file the dashboard reads, as a dict of {file stem: content}."""
    h, train = policies[policies["is_holdout"]], policies[~policies["is_holdout"]]
    comparison = saved["model_comparison"]
    final = saved["metrics_final"]

    age = rates_by(h, "DrivAge", {"glm": "freq_glm_final", "gbm": "freq_gbm_final"}, *AGE_RANGE)
    age["DrivAge"] = age["DrivAge"].astype(int)

    shap = saved["metrics_interactions"]["shap"]
    importance = [{"factor": FACTOR_NAMES[k], "mean_abs_shap": v} for k, v in shap["mean_abs_shap_by_factor"].items()]

    ratio = h["pp_gbm_final"] / h["pp_glm_final"]
    lift = lift_by_average(h["freq_glm_final"], h["freq_gbm_final"], h["ClaimNb"], h["Exposure"], LIFT_GROUPS)
    lift = lift.rename(columns={"rate_a": "rate_glm", "rate_b": "rate_gbm"})

    lorenz = {}
    for key in ["glm_final", "gbm_final"]:
        x, y = lorenz_points(h[f"freq_{key}"], h["ClaimNb"], h["Exposure"], LORENZ_POINTS)
        lorenz[key] = {"x": x.tolist(), "y": y.tolist(), "gini": final[key]["holdout"]["frequency"]["gini"]}

    density = train["Density"].to_numpy(dtype=float)
    grid = np.geomspace(np.percentile(density, 1), np.percentile(density, 99), 40)

    classes = pd.Series(exposure_class(h["Exposure"]), index=h.index)
    record_length = []
    for k in ["under half a year", "half to one year", "full year"]:
        m = classes == k
        record_length.append({"record_length": k, "exposure_share": float(h.loc[m, "Exposure"].sum() / h["Exposure"].sum()),
                              "claims": int(h.loc[m, "ClaimNb"].sum()),
                              "a_over_e_glm": final["glm_final"]["holdout"]["frequency_a_over_e_by_exposure_class"][k],
                              "a_over_e_gbm": final["gbm_final"]["holdout"]["frequency_a_over_e_by_exposure_class"][k]})

    return {
        "models": {"holdout": comparison["holdout"], "train": comparison["train"], "rows": model_rows(saved)},
        "driver_age": {"data": "holdout", "rows": age.rename(columns={"DrivAge": "age"}).to_dict(orient="records")},
        "shap": {"sample": shap["sample"], "rows_used": shap["rows_for_importance"], "rows": importance},
        "score_rates": {"data": "training rows, full-year records",
                        "rows": claim_rate_by_score(train, *SCORE_RANGE, MIN_POLICIES_PER_SCORE).to_dict(orient="records")},
        "price_ratio": {"data": "holdout", **price_ratio_summary(ratio, h["Exposure"], PRICE_RATIO_EDGES, PRICE_THRESHOLD)},
        "double_lift": {"data": "holdout", "rows": comparison["final"]["double_lift"]},
        "lift": {"data": "holdout, final models, sorted by the average of the two predictions",
                 "rows": lift.to_dict(orient="records")},
        "lorenz": {"data": "holdout", **lorenz},
        "relativities": glm_relativities(saved["final_glm"]["frequency"]["coefficients"], grid, float(np.median(density))),
        "record_length": {"data": "holdout", "rows": record_length},
        "timing_checks": timing_checks(saved["bonus_malus_timing"]),
    }


def main():
    names = ["model_comparison", "metrics_glm", "metrics_gbm", "metrics_interactions", "metrics_final",
             "final_glm", "bonus_malus_timing"]
    saved = {n: json.loads((OUTPUTS / f"{n}.json").read_text(encoding="utf-8")) for n in names}
    files = build(pd.read_parquet(PREDICTIONS_PATH), saved)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    for stem, content in files.items():
        path = DASHBOARD_DIR / f"{stem}.json"
        path.write_text(json.dumps(content, indent=1), encoding="utf-8")
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
