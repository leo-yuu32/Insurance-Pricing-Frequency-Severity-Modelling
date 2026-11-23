"""Evidence for the GLM design choices, using cross-validation on the TRAINING data only.

Prints the comparison tables. Nothing here touches the holdout.

Run from the repo root:  python src/compare_glm_designs.py
"""

import pandas as pd

import data
from design import FREQUENCY_TERMS, add_model_columns
from glm import cv_frequency_deviance, cv_severity_deviance


def swap(terms, old, new):
    return [new if t == old else t for t in terms]


def main():
    policies, _, _ = data.load_dataset()
    train = add_model_columns(policies[~policies["is_holdout"]])

    print("Frequency GLM: 4-fold cross-validated Poisson deviance per unit exposure (lower is better)")
    variants = {
        "baseline (FREQUENCY_TERMS)": FREQUENCY_TERMS,
        "BonusMalus linear, capped at 150": swap(FREQUENCY_TERMS, "BonusMalusBand", "BonusMalusCapped"),
        "Area in place of log Density": swap(FREQUENCY_TERMS, "LogDensity", "Area"),
        "Area as well as log Density": FREQUENCY_TERMS + ["Area"],
        "legacy 8 driver-age bands": swap(FREQUENCY_TERMS, "DrivAgeBand", "DrivAgeBand8"),
        "legacy factor set (no BonusMalus, no Density)": ["DrivAgeBand8", "VehAgeBand", "VehPowerBand", "VehBrand", "VehGas", "Region", "Area"],
        "intercept only": [],
    }
    rows = [{"design": name, "cv_deviance": cv_frequency_deviance(train, terms)} for name, terms in variants.items()]
    table = pd.DataFrame(rows)
    table["vs_baseline"] = table["cv_deviance"] - table["cv_deviance"].iloc[0]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    claims = train[train["ClaimRecords"] > 0]
    print(f"\nSeverity GLM: 4-fold cross-validated Gamma deviance per claim, {len(claims)} training policies with claims")
    full = FREQUENCY_TERMS
    rows = [{"design": "all eight terms", "cv_deviance": cv_severity_deviance(claims, full)},
            {"design": "intercept only", "cv_deviance": cv_severity_deviance(claims, [])}]
    for term in full:
        rows.append({"design": f"drop {term}", "cv_deviance": cv_severity_deviance(claims, [t for t in full if t != term])})
        rows.append({"design": f"only {term}", "cv_deviance": cv_severity_deviance(claims, [term])})
    table = pd.DataFrame(rows)
    table["vs_intercept_only"] = table["cv_deviance"] - table["cv_deviance"].iloc[1]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
