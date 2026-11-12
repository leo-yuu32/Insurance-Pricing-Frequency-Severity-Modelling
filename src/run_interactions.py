"""Explain the GBM with SHAP and feed what it shows back into the GLM.

Run from the repo root, after run_gbm.py:
    python src/run_interactions.py

Steps
  1. Refit the tuned GBM on the training rows.
  2. SHAP values and SHAP interaction values on TRAINING samples: factor importance, and a
     ranking of pairs of factors (candidates only).
  3. Candidate GLM terms are scored by 4-fold cross-validated Poisson deviance on the training
     data. Two kinds of term come out of the SHAP analysis:
       - interactions, from the SHAP interaction ranking;
       - the bonus-malus "off the claim-free ladder" indicator, from the SHAP dependence plot.
  4. Three extended GLMs are refitted on all training rows and scored once on the holdout:
       glm_int     GLM + the best interaction
       glm_ladder  GLM + the ladder indicator
       glm_both    GLM + the ladder indicator + the best interaction given the indicator

Writes outputs/metrics_interactions.json, outputs/glm_*_summary.txt, plots/shap_*.png and adds
the new prediction columns to outputs/predictions.parquet.
"""

import gc
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy.stats import chi2  # noqa: E402

import gbm  # noqa: E402
import metrics  # noqa: E402
from config import SEED  # noqa: E402
from design import FREQUENCY_TERMS, Design, add_model_columns  # noqa: E402
from glm import cv_frequency_deviance, fit_poisson, predict  # noqa: E402
from run_gbm import PLOTS  # noqa: E402
from run_glm import OUTPUTS, PREDICTIONS_PATH, holdout_metrics, rebalance_factor, summary_text  # noqa: E402

N_SHAP_ROWS = 10000
N_INTERACTION_ROWS = 2000
N_CANDIDATES = 5
MAX_EXTRA_PARAMETERS = 50  # an interaction needing more columns than this is not a usable rating term
LADDER_TERM = "BonusMalusOffLadder"
NEW_MODELS = ["glm_int", "glm_ladder", "glm_both"]
# The GLM term used when a raw factor takes part in an interaction: coarse bands keep the
# number of extra parameters small and the cells well populated.
INTERACTION_TERM = {
    "DrivAge": "DrivAgeCoarse", "BonusMalus": "BonusMalusCoarse", "VehAge": "VehAgeCoarse", "VehPower": "VehPowerCoarse",
    "Density": "LogDensity", "Area": "LogDensity", "VehBrand": "VehBrand", "VehGas": "VehGas", "Region": "Region",
}


def rank_interactions(explainer, X):
    """Pairs of features ordered by mean absolute SHAP interaction value (both halves of the pair)."""
    values = explainer.shap_interaction_values(X)
    strength = 2.0 * np.abs(values).mean(axis=0)
    pairs = []
    for i in range(len(gbm.FEATURES)):
        for j in range(i + 1, len(gbm.FEATURES)):
            pairs.append({"a": gbm.FEATURES[i], "b": gbm.FEATURES[j], "mean_abs_interaction": float(strength[i, j])})
    main = {gbm.FEATURES[i]: float(np.abs(values[:, i, i]).mean()) for i in range(len(gbm.FEATURES))}
    return sorted(pairs, key=lambda p: -p["mean_abs_interaction"]), main


def save_shap_plots(explainer, model, holdout, importance):
    """Summary and dependence plots, on a holdout sample (display only, not used for any choice)."""
    sample = holdout.sample(N_SHAP_ROWS, random_state=SEED)
    X = gbm.make_features(sample, model["levels"])
    values = explainer.shap_values(X)
    PLOTS.mkdir(exist_ok=True)
    shap.summary_plot(values, X, show=False)
    plt.gcf().savefig(PLOTS / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    for feature in list(importance)[:3]:
        shap.dependence_plot(feature, values, X, show=False)
        plt.gcf().savefig(PLOTS / f"shap_dependence_{feature}.png", dpi=150, bbox_inches="tight")
        plt.close("all")


def candidate_terms(ranking, train):
    """The top SHAP pairs as distinct pairs of GLM terms. Returns (candidates, skipped as too large)."""

    def n_columns(term):
        return 1 if pd.api.types.is_numeric_dtype(train[term]) else train[term].nunique() - 1

    out, skipped, seen = [], [], set()
    for pair in ranking:
        terms = (INTERACTION_TERM[pair["a"]], INTERACTION_TERM[pair["b"]])
        if terms[0] == terms[1] or frozenset(terms) in seen:
            continue
        seen.add(frozenset(terms))
        row = {"factors": [pair["a"], pair["b"]], "glm_terms": list(terms), "mean_abs_shap_interaction": pair["mean_abs_interaction"]}
        if n_columns(terms[0]) * n_columns(terms[1]) > MAX_EXTRA_PARAMETERS:
            skipped.append({**row, "extra_parameters": n_columns(terms[0]) * n_columns(terms[1])})
            continue
        out.append(row)
        if len(out) == N_CANDIDATES:
            break
    return out, skipped


def full_fit(train, terms, interactions=()):
    design = Design(terms, interactions).fit(train)
    return design, fit_poisson(design.transform(train), train["ClaimNb"], train["Exposure"])


def score_additions(train, base_terms, candidates):
    """Cross-validated deviance and a likelihood-ratio test for each candidate added to base_terms."""
    base_cv = cv_frequency_deviance(train, base_terms)
    base_design, base_fit = full_fit(train, base_terms)
    rows = []
    for cand in candidates:
        pair = tuple(cand["glm_terms"])
        design, fit = full_fit(train, base_terms, [pair])
        extra = len(design.columns) - len(base_design.columns)
        lr = float(2.0 * (fit.llf - base_fit.llf))
        cv = cv_frequency_deviance(train, base_terms, [pair])
        rows.append({**cand, "extra_parameters": extra, "cv_deviance": cv, "cv_deviance_reduction": base_cv - cv,
                     "lr_statistic": lr, "lr_p_value": float(chi2.sf(lr, extra))})
        del design, fit
        gc.collect()
        print(rows[-1], flush=True)
    return base_cv, base_fit, len(base_design.columns), rows


def main():
    saved = pd.read_parquet(PREDICTIONS_PATH)
    saved = saved.drop(columns=[c for c in saved.columns if c.endswith(tuple(NEW_MODELS))])
    policies = add_model_columns(saved)
    train, holdout = policies[~policies["is_holdout"]], policies[policies["is_holdout"]]
    null_rate = float(train["ClaimNb"].sum() / train["Exposure"].sum())

    # 1. the GBM being explained (same parameters and rows as run_gbm.py)
    best = json.loads((OUTPUTS / "gbm_tuning.json").read_text())["best"]
    model = gbm.fit(train, best["params"], best["n_rounds"])
    explainer = shap.TreeExplainer(model["booster"])

    # 2. SHAP on TRAINING samples
    X_imp = gbm.make_features(train.sample(N_SHAP_ROWS, random_state=SEED), model["levels"])
    shap_values = explainer.shap_values(X_imp)
    raw = model["booster"].predict(X_imp, raw_score=True)
    additivity_error = float(np.abs(shap_values.sum(axis=1) + explainer.expected_value - raw).max())
    importance = dict(sorted(zip(gbm.FEATURES, np.abs(shap_values).mean(axis=0).tolist()), key=lambda kv: -kv[1]))
    X_int = gbm.make_features(train.sample(N_INTERACTION_ROWS, random_state=SEED + 1), model["levels"])
    ranking, main_effects = rank_interactions(explainer, X_int)
    save_shap_plots(explainer, model, holdout, importance)
    candidates, skipped = candidate_terms(ranking, train)

    # 3. selection by cross-validation on the training data
    ladder_terms = FREQUENCY_TERMS + [LADDER_TERM]
    base_cv, base_fit, base_cols, on_glm = score_additions(train, FREQUENCY_TERMS, candidates)
    ladder_cv, ladder_fit, ladder_cols, on_ladder = score_additions(train, ladder_terms, candidates)
    best_int = max(on_glm, key=lambda c: c["cv_deviance_reduction"])
    best_int_given_ladder = max(on_ladder, key=lambda c: c["cv_deviance_reduction"])
    ladder_lr = float(2.0 * (ladder_fit.llf - base_fit.llf))
    result = {
        "shap": {"rows_for_importance": N_SHAP_ROWS, "rows_for_interactions": N_INTERACTION_ROWS, "sample": "training rows",
                 "additivity_max_error": additivity_error, "mean_abs_shap_by_factor": importance,
                 "mean_abs_main_effect_by_factor": main_effects, "interaction_ranking": ranking[:12]},
        "selection": {
            "criterion": "4-fold cross-validated Poisson deviance on training data",
            "glm_cv_deviance": base_cv, "skipped_as_too_large": skipped,
            "interactions_added_to_glm": on_glm, "chosen_interaction": best_int,
            "ladder_term": {"term": LADDER_TERM, "extra_parameters": ladder_cols - base_cols, "cv_deviance": ladder_cv,
                            "cv_deviance_reduction": base_cv - ladder_cv, "lr_statistic": ladder_lr,
                            "lr_p_value": float(chi2.sf(ladder_lr, ladder_cols - base_cols))},
            "interactions_added_to_glm_with_ladder": on_ladder, "chosen_interaction_given_ladder": best_int_given_ladder,
        },
        "models": {},
    }

    # 4. refit the three extended GLMs and score them on the holdout
    specs = {"glm_int": (FREQUENCY_TERMS, [tuple(best_int["glm_terms"])]),
             "glm_ladder": (ladder_terms, []),
             "glm_both": (ladder_terms, [tuple(best_int_given_ladder["glm_terms"])])}
    for name, (terms, interactions) in specs.items():
        design, fit = full_fit(train, terms, interactions)
        (OUTPUTS / f"{name}_summary.txt").write_text(summary_text(fit, design.columns), encoding="utf-8")
        policies[f"freq_{name}"] = predict(fit, design.transform(policies))
        t = policies[~policies["is_holdout"]]
        pp_factor = rebalance_factor(t["TotalLossCapped"], t[f"freq_{name}"] * t["sev_glm"], t["Exposure"])
        policies[f"pp_{name}"] = policies[f"freq_{name}"] * policies["sev_glm"] * pp_factor
        result["models"][name] = {
            "terms": terms, "interactions": [list(i) for i in interactions], "frequency_parameters": len(design.columns),
            "frequency_rebalance_factor": rebalance_factor(t["ClaimNb"], t[f"freq_{name}"], t["Exposure"]),
            "pure_premium_rebalance_factor": pp_factor,
            "holdout": holdout_metrics(policies, f"freq_{name}", null_rate, f"pp_{name}"),
        }

    h = policies[policies["is_holdout"]]
    compared = ["glm"] + NEW_MODELS + [m for m in ["gbm_additive", "gbm_monotone", "gbm"] if f"freq_{m}" in h]
    dev = {m: metrics.mean_poisson_deviance(h["ClaimNb"], h[f"freq_{m}"], h["Exposure"]) for m in compared}
    gini = {m: metrics.gini(h[f"freq_{m}"], h["ClaimNb"], h["Exposure"]) for m in compared}

    def boot(a, b):
        est, low, high = metrics.bootstrap_deviance_difference(h["ClaimNb"], h[f"freq_{a}"], h[f"freq_{b}"], h["Exposure"], n_boot=1000, seed=SEED)
        return {"deviance_reduction": est, "bootstrap_95_low": low, "bootstrap_95_high": high}

    gap = dev["glm"] - dev["gbm"]
    result["gap"] = {"holdout_deviance": dev, "holdout_gini": gini, "glm_to_gbm": boot("glm", "gbm")}
    for name in NEW_MODELS:
        result["gap"][name] = {
            "glm_to_model": boot("glm", name), "model_to_gbm": boot(name, "gbm"),
            "share_of_gap_closed_deviance": (dev["glm"] - dev[name]) / gap if gap > 0 else None,
            "share_of_gap_closed_gini": (gini[name] - gini["glm"]) / (gini["gbm"] - gini["glm"]) if gini["gbm"] > gini["glm"] else None,
        }
    if "gbm_additive" in dev:
        result["gap"]["glm_to_additive_gbm"] = boot("glm", "gbm_additive")
        result["gap"]["additive_gbm_to_gbm"] = boot("gbm_additive", "gbm")
        part = dev["gbm_additive"] - dev["gbm"]
        result["gap"]["glm_int"]["share_of_interaction_portion_closed"] = (dev["glm"] - dev["glm_int"]) / part if part > 0 else None
        result["gap"]["glm_both"]["share_of_interaction_portion_closed_by_the_interaction"] = (dev["glm_ladder"] - dev["glm_both"]) / part if part > 0 else None

    keep = [c for c in policies.columns if c in saved.columns or c.endswith(tuple(NEW_MODELS))]
    policies[keep].to_parquet(PREDICTIONS_PATH, index=False)
    (OUTPUTS / "metrics_interactions.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["gap"], indent=2))
    print("importance:", importance)
    print("top pairs:", [(p["a"], p["b"], round(p["mean_abs_interaction"], 4)) for p in ranking[:6]])


if __name__ == "__main__":
    main()
