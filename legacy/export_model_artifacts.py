"""
export_model_artifacts.py

Fetching variables for eExcel cross-validation workstream.

Trains the full pricing pipeline (via run_pricing_pipeline) and serialises
all model artifacts needed to replicate predictions in Excel:

  - frequency_coefficients.csv    Poisson GLM intercept + betas
  - severity_coefficients.csv     Gamma GLM intercept + betas
  - encoder_reference_levels.csv  Reference categories for OneHotEncoder(drop='first')
  - isotonic_knots.csv            X_thresholds_, y_thresholds_ from calibrator
  - loadings.csv                  Flat version of config.ACTUARIAL_LOADINGS
  - scalars.json                  Off-balance factor + key hyperparameters
  - test_sample.csv               100 test policies with Python predictions

Run from project root:
    python src/export_model_artifacts.py
"""

import json
from pathlib import Path

import pandas as pd

import config
from pricing_engine import run_pricing_pipeline


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def ensure_artifacts_dir():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def extract_coefficients(model_pipeline, model_label):
    """
    Pulls intercept + coefficients from a fitted sklearn Pipeline whose final
    step is a GLM (PoissonRegressor / GammaRegressor) with a OneHotEncoder in
    the preprocessor.

    Returns a long-format DataFrame: one row per encoded dummy variable plus
    a single intercept row. This format drops straight into Excel as the
    lookup table for a SUMPRODUCT-based rating formula.
    """
    preprocessor = model_pipeline.named_steps["preprocessor"]
    glm = model_pipeline.named_steps["regressor"]

    # get_feature_names_out returns names like 'cat__VehGas_Diesel' that line
    # up exactly with the column order of glm.coef_. Strip the 'cat__' prefix
    # for cleaner Excel display.
    encoded_names = preprocessor.get_feature_names_out()

    if len(encoded_names) != len(glm.coef_):
        raise ValueError(
            f"Coefficient/name length mismatch for {model_label}: "
            f"{len(encoded_names)} names vs {len(glm.coef_)} coefs"
        )

    rows = [{"encoded_feature": "_intercept_", "coefficient": float(glm.intercept_)}]
    for name, coef in zip(encoded_names, glm.coef_):
        clean_name = name.split("__", 1)[1] if "__" in name else name
        rows.append({"encoded_feature": clean_name, "coefficient": float(coef)})

    return pd.DataFrame(rows)


def extract_reference_levels(model_pipeline):
    """
    For each categorical feature, records the level dropped by
    OneHotEncoder(drop='first'). This is essential for Excel: a policy whose
    feature value equals the reference level contributes 0 to SUMPRODUCT for
    that feature (no dummy active = baseline risk for that factor).
    """
    preprocessor = model_pipeline.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["cat"]

    rows = []
    for feature, categories in zip(config.CATEGORICAL_FEATURES, encoder.categories_):
        categories = [str(c) for c in categories]
        rows.append(
            {
                "feature": feature,
                "reference_level": categories[0],
                "non_reference_levels": "|".join(categories[1:]),
            }
        )
    return pd.DataFrame(rows)


def extract_isotonic_knots(calibrator):
    """
    sklearn's IsotonicRegression stores its piecewise linear breakpoints as
    X_thresholds_ / y_thresholds_. Excel replicates the transform with linear
    interpolation between consecutive knots (FORECAST.LINEAR works here, or a
    small LET-based formula).
    """
    return pd.DataFrame(
        {
            "x": calibrator.X_thresholds_,
            "y": calibrator.y_thresholds_,
        }
    )


def extract_loadings():
    """
    Flatten the nested ACTUARIAL_LOADINGS dict from config.py into a long
    CSV. Excel will VLOOKUP into this on a concatenated (feature, bin) key.
    """
    rows = []
    for feature, rules in config.ACTUARIAL_LOADINGS.items():
        for bin_value, multiplier in rules.items():
            rows.append(
                {
                    "feature": feature,
                    "bin": str(bin_value),
                    "multiplier": float(multiplier),
                }
            )
    return pd.DataFrame(rows)


def extract_test_sample(test_results, n=100, seed=42):
    """
    Pick 100 random test policies with all the inputs Excel needs plus
    Python's prediction at every stage of the pipeline. Phase 3 reconciliation
    runs against these.
    """
    # Only DriverAge, VehAge, VehPower, VehBrand have pre-binned originals
    # worth keeping for sense-checking. VehGas, Region, Area aren't binned, so
    # they appear only in CATEGORICAL_FEATURES (avoiding duplicate columns).
    cols = [
        "DriverAge", "VehAge", "VehPower", "VehBrand",
        *config.CATEGORICAL_FEATURES,
        "Exposure", "ClaimNb", "TotalLoss",
        # Python predictions at each stage - the reconciliation targets
        "PredFreq", "PredSev",
        "PurePremium_Raw", "PurePremium_OB",
        "PurePremium_Calibrated", "Final_Price",
    ]
    # Defensive: only keep columns that actually exist + dedupe
    seen = set()
    cols = [c for c in cols if c in test_results.columns and not (c in seen or seen.add(c))]
    sample = test_results.sample(n=n, random_state=seed)[cols].copy()
    sample.reset_index(drop=False, inplace=True)
    sample.rename(columns={"index": "original_index"}, inplace=True)
    return sample


def main():
    out_dir = ensure_artifacts_dir()
    print(f"Writing artifacts to: {out_dir}/\n")

    print("Running full pipeline (this may take a few minutes)...")
    result = run_pricing_pipeline(return_train_data=True)
    models = result["models"]
    test_results = result["test_results"]

    # --- Coefficients ---
    freq_coefs = extract_coefficients(models["freq_m"], "frequency")
    freq_coefs.to_csv(out_dir / "frequency_coefficients.csv", index=False)
    print(f"  frequency_coefficients.csv     ({len(freq_coefs)} rows)")

    sev_coefs = extract_coefficients(models["sev_m"], "severity")
    sev_coefs.to_csv(out_dir / "severity_coefficients.csv", index=False)
    print(f"  severity_coefficients.csv      ({len(sev_coefs)} rows)")

    # --- Reference levels ---
    # Both models share the same CATEGORICAL_FEATURES spec, but they're fitted
    # on different subsets (severity model trains on ClaimNb>0 only), so their
    # encoder.categories_ could in principle differ. Frequency model uses the
    # full training set, so we treat that as the canonical encoder map.
    ref_levels = extract_reference_levels(models["freq_m"])
    ref_levels.to_csv(out_dir / "encoder_reference_levels.csv", index=False)
    print(f"  encoder_reference_levels.csv   ({len(ref_levels)} rows)")

    # --- Isotonic knots ---
    knots = extract_isotonic_knots(models["calibrator"])
    knots.to_csv(out_dir / "isotonic_knots.csv", index=False)
    print(f"  isotonic_knots.csv             ({len(knots)} knots)")

    # --- Loadings ---
    loadings = extract_loadings()
    loadings.to_csv(out_dir / "loadings.csv", index=False)
    print(f"  loadings.csv                   ({len(loadings)} rows)")

    # --- Scalars ---
    scalars = {
        "off_balance_factor": float(result["off_balance_factor"]),
        "gini_technical": float(result["gini_technical"]),
        "gini_commercial": float(result["gini_commercial"]),
        "avg_premium": float(result["avg_premium"]),
        "calibration_cap_threshold": config.CALIBRATION_CAP_THRESHOLD,
        "severity_cap_percentile": config.SEVERITY_CAP_PERCENTILE,
        "test_size": config.TEST_SIZE,
        "random_state": config.RANDOM_STATE,
        "glm_alpha": config.GLM_ALPHA,
    }
    with open(out_dir / "scalars.json", "w") as f:
        json.dump(scalars, f, indent=2)
    print(f"  scalars.json                   ({len(scalars)} entries)")

    # --- Test sample ---
    sample = extract_test_sample(test_results, n=100, seed=42)
    sample.to_csv(out_dir / "test_sample.csv", index=False)
    print(f"  test_sample.csv                ({len(sample)} policies)")

    # --- Quick sanity printout ---
    print("\n--- Sanity check: first sample policy ---")
    row = sample.iloc[0]
    print(f"  original_index : {row['original_index']}")
    print(
        f"  DriverAge_Bin={row['DriverAge_Bin']}, "
        f"VehAge_Bin={row['VehAge_Bin']}, "
        f"VehPower_Bin={row['VehPower_Bin']}"
    )
    print(f"  PredFreq               = {row['PredFreq']:.6f}")
    print(f"  PredSev                = EUR {row['PredSev']:.4f}")
    print(f"  PurePremium_OB         = EUR {row['PurePremium_OB']:.4f}")
    print(f"  PurePremium_Calibrated = EUR {row['PurePremium_Calibrated']:.4f}")
    print(f"  Final_Price            = EUR {row['Final_Price']:.4f}")
    print("\nDone. These artifacts are the inputs to the Excel rating engine (Phase 2).")


if __name__ == "__main__":
    main()
