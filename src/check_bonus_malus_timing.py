"""Is BonusMalus recorded before the claims it is used to predict?

freMTPL2 does not document this. If the recorded coefficient already reflects claims that are
in the claim count, it is partly an outcome and not a rating factor (target leakage). This
script collects the evidence, on TRAINING rows only. Read it with docs/DECISIONS.md, D30.

Run from the repo root, after run_glm.py (about 6 minutes):
    python src/check_bonus_malus_timing.py

Writes outputs/bonus_malus_timing.json.
"""

import json

import numpy as np
import pandas as pd

import gbm
from design import CLAIM_FREE_LADDER, FREQUENCY_TERMS, Design, add_model_columns
from glm import cv_frequency_deviance, fit_poisson
from metrics import poisson_deviance
from run_glm import OUTPUTS, PREDICTIONS_PATH

# A driver at 50 who is surcharged for a claim goes to 62 (50 x 1.25, truncated). Each claim-free
# year after that multiplies by 0.95: 62 -> 58 -> 55 -> 52 -> 50. So among these values, the
# value says how many renewals have passed since the surcharge.
CHAIN_FROM_50 = {62: "62 (surcharged at the last renewal)", 58: "58 (one renewal later)",
                 55: "55 (two renewals later)", 52: "52 (three renewals later)"}
# The values one full surcharge above a claim-free value, where that is not itself claim-free.
ONE_SURCHARGE_ABOVE_LADDER = sorted({v * 125 // 100 for v in CLAIM_FREE_LADDER} - set(CLAIM_FREE_LADDER))
N_BLOCKS = 8


def value_group(bonus_malus):
    """Label a bonus-malus value for the tables below."""
    v = int(bonus_malus)
    if v == 50:
        return "50"
    if v in CHAIN_FROM_50:
        return CHAIN_FROM_50[v]
    if v in (51, 54, 57):
        return "51, 54, 57 (claim-free values next to 50)"
    if v in ONE_SURCHARGE_ABOVE_LADDER:
        return "one surcharge above another claim-free value"
    return "other"


def exposure_class(exposure):
    exposure = np.asarray(exposure, dtype=float)
    return np.where(exposure >= 0.999, "full year", np.where(exposure >= 0.5, "half to one year", "under half a year"))


def frequency_table(df, by):
    """Policies, exposure, claims, claims per policy-year and the share of policies with a claim."""
    g = df.groupby(by, observed=True).agg(
        policies=("ClaimNb", "size"), exposure=("Exposure", "sum"), claims=("ClaimNb", "sum"),
        share_with_claim=("ClaimNb", lambda s: float((s > 0).mean())))
    g["frequency"] = g["claims"] / g["exposure"]
    return g


def indicator_by_exposure_class(train):
    """The off-ladder relativity from the GLM, fitted separately for each exposure class."""
    m = train.copy()
    extra = []
    for label in ["full year", "half to one year", "under half a year"]:
        here = (m["ExposureClass"] == label).astype(float)
        if label != "under half a year":
            m[f"Class[{label}]"] = here
            extra.append(f"Class[{label}]")
        m[f"OffLadder[{label}]"] = m["BonusMalusOffLadder"] * here
        extra.append(f"OffLadder[{label}]")
    design = Design(FREQUENCY_TERMS + extra).fit(m)
    fit = fit_poisson(design.transform(m), m["ClaimNb"], m["Exposure"])
    params = pd.Series(np.asarray(fit.params), index=design.columns)
    return {c: float(np.exp(params[c])) for c in extra if c.startswith("OffLadder")}


def cv_null(train):
    total = 0.0
    for k in sorted(train["fold"].unique()):
        fit_rows, test_rows = train[train["fold"] != k], train[train["fold"] == k]
        rate = fit_rows["ClaimNb"].sum() / fit_rows["Exposure"].sum()
        total += poisson_deviance(test_rows["ClaimNb"], rate * test_rows["Exposure"])
    return total / float(train["Exposure"].sum())


def cv_with_and_without_bonus_malus(train):
    """Cross-validated deviance on training data. The GBM keeps its tuned parameters (not re-tuned)."""
    tuned = json.loads((OUTPUTS / "gbm_tuning.json").read_text())["best"]["params"]
    out = {"null": cv_null(train),
           "glm": cv_frequency_deviance(train, FREQUENCY_TERMS),
           "glm_without_bonus_malus": cv_frequency_deviance(train, [t for t in FREQUENCY_TERMS if t != "BonusMalusBand"]),
           "gbm": gbm.cv_score(train, tuned)[0]}
    all_features = gbm.FEATURES
    try:  # gbm.py reads its feature list from this module-level name
        gbm.FEATURES = [f for f in all_features if f != "BonusMalus"]
        out["gbm_without_bonus_malus"] = gbm.cv_score(train, tuned)[0]
    finally:
        gbm.FEATURES = all_features
    out["glm_to_gbm_gap"] = out["glm"] - out["gbm"]
    out["glm_to_gbm_gap_without_bonus_malus"] = out["glm_without_bonus_malus"] - out["gbm_without_bonus_malus"]
    return out


def as_records(table):
    return json.loads(table.reset_index().to_json(orient="records"))


def main():
    saved = pd.read_parquet(PREDICTIONS_PATH)
    train = add_model_columns(saved[~saved["is_holdout"]])
    train["HasClaim"] = train["ClaimNb"] > 0
    train["OffLadder"] = train["BonusMalusOffLadder"] == 1.0
    train["ExposureClass"] = exposure_class(train["Exposure"])
    train["ValueGroup"] = train["BonusMalus"].map(value_group)
    train["IDpolBlock"] = pd.qcut(train["IDpol"], N_BLOCKS).astype(str)
    block_order = train.groupby("IDpolBlock")["IDpol"].min().sort_values().index
    train["IDpolBlock"] = train["IDpolBlock"].map({b: f"{i + 1} (IDpol from {int(train.loc[train['IDpolBlock'] == b, 'IDpol'].min()):,})"
                                                   for i, b in enumerate(block_order)})
    below_100 = train[train["BonusMalus"] < 100]
    full_year = train[train["ExposureClass"] == "full year"]

    result = {
        "rows": "training rows only", "policies": int(len(train)),
        "one_surcharge_above_ladder": ONE_SURCHARGE_ABOVE_LADDER,
        "off_ladder_share": {
            "policies_with_a_claim": float(train.loc[train["HasClaim"], "OffLadder"].mean()),
            "policies_without_a_claim": float(train.loc[~train["HasClaim"], "OffLadder"].mean()),
            "off_ladder_policies_with_no_claim": float(1.0 - train.loc[train["OffLadder"], "HasClaim"].mean()),
        },
        "by_exposure_class_below_100": as_records(frequency_table(below_100, ["ExposureClass", "OffLadder"])),
        "glm_off_ladder_relativity_by_exposure_class": indicator_by_exposure_class(train),
        "full_year_records_by_value": as_records(frequency_table(full_year, "ValueGroup")),
        "by_idpol_block_and_value": as_records(frequency_table(train[train["ValueGroup"] != "other"], ["IDpolBlock", "ValueGroup"])),
        "idpol_blocks": as_records(train.groupby("IDpolBlock").agg(policies=("Exposure", "size"), mean_exposure=("Exposure", "mean"),
                                                                   full_year_share=("ExposureClass", lambda s: float((s == "full year").mean())))),
    }
    result["cross_validated_deviance"] = cv_with_and_without_bonus_malus(train)
    (OUTPUTS / "bonus_malus_timing.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    pd.set_option("display.width", 220)
    print(json.dumps(result["off_ladder_share"], indent=2))
    print(pd.DataFrame(result["by_exposure_class_below_100"]).round(4).to_string(index=False))
    print(json.dumps(result["glm_off_ladder_relativity_by_exposure_class"], indent=2))
    print(pd.DataFrame(result["full_year_records_by_value"]).round(4).to_string(index=False))
    wide = pd.DataFrame(result["by_idpol_block_and_value"]).pivot(index="IDpolBlock", columns="ValueGroup", values="frequency")
    print(wide.round(3).to_string())
    print(pd.DataFrame(result["by_idpol_block_and_value"]).pivot(index="IDpolBlock", columns="ValueGroup", values="claims").to_string())
    print(pd.DataFrame(result["idpol_blocks"]).round(3).to_string(index=False))
    print(json.dumps(result["cross_validated_deviance"], indent=2))


if __name__ == "__main__":
    main()
