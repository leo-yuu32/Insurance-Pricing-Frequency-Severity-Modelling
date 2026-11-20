# CV summary: fact sheet

A fact sheet for whoever writes the CV, not CV wording. Every number comes from the code's outputs (`outputs/*.json`, the test suite and `docs/DECISIONS.md`), as of 20 November 2026. Only work marked done may go on the CV.

## 1. Status

| Phase | Status |
|---|---|
| Phase 0: audit of the original project | Done (`docs/AUDIT.md`) |
| Phase 1: data conventions, split, metrics, statsmodels GLM | Done |
| Phase 2: NumPy IRLS solver validated against statsmodels | Done |
| Phase 3: LightGBM frequency model against the GLM | Done |
| Phase 4: SHAP, and one interaction fed back into the GLM | Done. The interaction is not in the final model (it did not improve the holdout significantly), and the Phase 4 result led to the `BonusMalus` timing finding below. |
| Stretch: additive GBM, bootstrap intervals | Done. The additive GBM was built with all nine factors only. |
| Phase 5: final model, dashboard, Excel engine, README, interview notes | Done on a local branch. **Not pushed or deployed**: the live app still runs the original version. |

## 2. Project in two sentences

A claim frequency and severity pricing model for French motor third-party liability insurance (the public freMTPL2 dataset, 668,626 policies after cleaning), comparing a Poisson GLM fitted with statsmodels against a LightGBM model on the same data, factors, split and metrics. It includes a from-scratch NumPy GLM solver validated against statsmodels, an Excel rating engine that reproduces the final GLM's prices, and a Streamlit dashboard reading precomputed results.

## 3. Data and validation setup

- **Dataset:** freMTPL2freq (678,013 policies) and freMTPL2sev (claim amounts), OpenML IDs 41214 and 41215.
- **Cleaning:**
  - excluded 9,387 policies with `IDpol` up to 24,951 (all have claims, almost none has claim amounts);
  - claim counts taken from the severity records and capped at 4 (8 policies affected);
  - exposure capped at 1 (1,224 policies);
  - individual claims capped at EUR 181,271, the 99.9th percentile of training claims (21 training claims capped). The excess is a flat large-loss load of 1.192.
- **Rows used:** 668,626 policies, 353,560 policy-years, 26,098 claims.
- **Split:** five folds with `StratifiedGroupKFold`, seed 42, stratified on whether a policy has a claim and grouped so that policies with identical rating factors stay on one side. Training: 534,900 policies (20,883 claims). Holdout: 133,726 policies (70,827 policy-years, 5,215 claims). Holdout IDs are saved in `artifacts/holdout_idpol.txt`.
- **Training data only** (4-fold cross-validation on the training folds): GLM banding and term choices, the severity factor, GBM hyperparameter tuning (30-trial random search with early stopping), SHAP values and interaction selection, and the `BonusMalus` timing checks.
- **Holdout:** scored for seven nine-factor models, then for the final pair once, after their specification was committed (`docs/DECISIONS.md`, D31). Two choices were made with earlier holdout results in view: dropping the interaction, and investigating `BonusMalus`, which a holdout result prompted, although the investigation itself used training data. The final figures may therefore be mildly optimistic.

## 4. Models

- **GLM (statsmodels, final):** Poisson, log link, log(exposure) offset, unpenalised; 55 parameters. Factors: driver age (10 bands), vehicle age (5 bands), vehicle power, log density, vehicle brand, fuel, region. `BonusMalus` excluded (see section 5). `Area` excluded as a banding of `Density`.
- **GLM with interactions:** not part of the final model. In Phase 4, on all nine factors, `BonusMalus` by `Region` (42 parameters) was selected by cross-validated deviance. It reduced holdout deviance by 0.00052 (95% interval -0.00001 to 0.00101), 8% of the gap at the time, which is not significant.
- **GBM (final):** LightGBM, Poisson objective, exposure as an offset through `init_score`; the same eight factors raw. Learning rate 0.1264, 17 leaves, minimum 1,791 policies per leaf, L2 penalty 13.36, 92 trees. No monotone constraint.
- **Additive GBM:** built with all nine factors only, as a diagnostic (each tree splits on one factor). Learning rate 0.1095, 11 leaves, 418 trees.
- **Severity:** one Gamma GLM (log link) on capped claims, driver age only, shared by every frequency model.
- **Calibration:** one scalar rebalancing factor per model, fitted on training data; all within 0.3% of 1. No isotonic calibration, no loadings.

## 5. Results table (holdout)

Gini definition: policies ordered from lowest to highest predicted claim frequency; Gini is twice the area between the Lorenz curve (cumulative exposure against cumulative claims) and the diagonal, so a model with no discrimination scores 0. Ties are pooled.

**Final models (eight factors, without `BonusMalus`):**

| Model | Poisson deviance | Frequency Gini | Actual / expected | Share of gap closed |
|---|---|---|---|---|
| GLM | 0.4687 | 0.179 | 1.00 | not applicable |
| GBM | 0.4683 | 0.183 | 1.00 | not applicable |

GLM minus GBM: 0.0003 (95% bootstrap interval -0.0002 to 0.0008). **Not significant.** No interaction was added to the final GLM, so there is no share of gap to report.

**Comparison with `BonusMalus` (nine factors).** Kept as the analysis that exposed the timing problem, not as pricing results:

| Model | Poisson deviance | Frequency Gini | Actual / expected | Share of gap closed |
|---|---|---|---|---|
| GLM | 0.4539 | 0.297 | 1.00 | |
| GLM + ladder indicator | 0.4499 | 0.315 | 1.00 | 66% |
| Additive GBM | 0.4488 | 0.319 | 1.00 | |
| GBM | 0.4478 | 0.320 | 1.00 | |

GLM minus GBM with `BonusMalus`: 0.0061 (interval 0.0049 to 0.0073).

**The `BonusMalus` finding:** the dataset does not document when the bonus-malus coefficient was recorded. Three checks on training data indicate it was recorded after some of the claims it is used to predict. The evidence is strong but unproven, so the final models exclude it (`docs/DECISIONS.md`, D30).

## 6. The Gini 0.43 to 0.48 history

- **What the figures were:** pure premium Ginis of two price columns of the original project on one test set: 0.4265 for the model's premium, 0.4812 after isotonic calibration and seven hand-set loadings.
- **What caused the change:** isotonic calibration +0.012, the driver age 18 to 21 loading (x2.40) +0.023, the other loadings +0.019. The loadings were read off charts of the same test set they were scored on.
- **Why they are retired:**
  - the Gini function ranked policies by premium divided by exposure, under which a constant prediction scored 0.444, higher than the model;
  - one claim of EUR 4,075,401 was 28.5% of test losses, and without it the 18 to 21 loading added nothing;
  - the data included the excluded `IDpol` block.

  Neither figure is a measure of model quality.
- **Replacement figures** (holdout, corrected definition, frequency Gini):
  - final GLM 0.179 and final GBM 0.183;
  - pure premium Gini on capped losses: 0.199 and 0.205.

## 7. Solver validation

- **What was compared:** the NumPy IRLS solver (`src/irls.py`) against statsmodels on coefficients, standard errors, deviance and, for Gamma, the dispersion. It uses statsmodels' convention, Pearson chi-squared over residual degrees of freedom.
- **Data:** synthetic data with known coefficients (Poisson with an offset, Gamma with prior weights) and the pipeline's real design matrices.
- **Agreement achieved (largest relative difference):**

| Design matrix | Coefficients | Standard errors | Notes |
|---|---|---|---|
| Final frequency GLM, 534,900 by 55 | 6.0e-12 | 9.1e-9 | deviance difference 2.9e-11 |
| Nine-factor frequency GLM, 534,900 by 63 | 2.6e-12 | 4.4e-9 | |
| Severity GLM, 19,721 by 10 | 1.1e-12 | 6.2e-14 | dispersion 13.0122399303 in both |

- **Test tolerance:** relative 1e-6 with absolute 1e-8.
- **Also tested:** one Poisson IRLS step equals a Newton-Raphson step (to 1e-10); the Gamma log-link step equals Fisher scoring and differs from Newton-Raphson.
- **Number of tests:** 8 in `tests/test_irls.py`, out of 60 in the suite, all passing.

## 8. Claims check

| Current CV claim | Status | What is accurate |
|---|---|---|
| "Built a frequency-severity pricing model (Poisson and Gamma GLMs) on the French Motor Third-Party Liability dataset" | Supported | As stated. The GLMs are now fitted with statsmodels. |
| "Capped large losses at the 99.9th percentile and applied isotonic calibration to align predicted and observed loss cost" | Needs rewording | Individual claims are capped at the 99.9th percentile of training claims, and the excess is a flat large-loss load. There is no isotonic calibration: each model has one rebalancing factor fitted on training data (within 0.3% of 1). |
| "Improved the Gini coefficient from 0.43 to 0.48 by [...]" | Unsupported, remove | See section 6. |
| "Deployed a Streamlit dashboard for dislocation analysis, using double-lift charts and Lorenz curves [...]; replicated core calculations in Excel to cross-validate the Python outputs" | Needs rewording | The rebuilt dashboard has a real GLM against GBM double-lift chart and a metrics table, but no dislocation analysis and no Lorenz curve, and it is **not yet deployed**. The live app is the original version, which shows single-model lift charts. Excel: the new workbook reproduces the final GLM's price for 100 of 100 holdout policies. |
| "Implemented a Poisson GLM with log link and exposure offset in NumPy, fitted by IRLS (equivalent to Newton-Raphson under the canonical link)" | Supported | It now also fits the Gamma log-link model (by Fisher scoring) and is validated against statsmodels on a 534,900 by 55 design matrix (coefficients to 6e-12 relative). |

## 9. Safe to claim

| Statement | Evidence |
|---|---|
| Built Poisson frequency and Gamma severity GLMs in statsmodels on 668,626 freMTPL2 policies, with exposure as an offset. | `src/run_final.py`, `outputs/metrics_final.json` |
| Compared a Poisson GLM with a LightGBM model on a 133,726-policy holdout using Poisson deviance, Gini, actual against expected and a double-lift chart, with paired bootstrap intervals. | `outputs/model_comparison.json`, `plots/double_lift_final_glm_vs_gbm.png` |
| On the eight factors other than `BonusMalus`, a tuned GBM did not significantly beat the GLM (deviance difference 0.0003, interval includes zero). | `outputs/model_comparison.json`, `docs/DECISIONS.md` D32 |
| Tuned LightGBM by grouped, stratified 4-fold cross-validation with early stopping on training data only. | `src/tune_gbm.py`, `outputs/gbm_tuning_final.json` |
| Used SHAP values and SHAP interaction values on the GBM to find candidate GLM terms. | `src/run_interactions.py`, `outputs/metrics_interactions.json` |
| Identified evidence of target leakage in the bonus-malus factor through three diagnostic checks, and excluded it from the final model. | `src/check_bonus_malus_timing.py`, `outputs/bonus_malus_timing.json`, `docs/DECISIONS.md` D30 |
| Fixed the final model's specification before scoring it on the holdout. | Commit `f954934` (specification) precedes commit `2f9c9da` (scores) |
| Wrote a GLM solver from scratch in NumPy (IRLS) for Poisson and Gamma families, validated against statsmodels to 6e-12 on a 534,900 by 55 design matrix. | `src/irls.py`, `tests/test_irls.py` |
| Built an Excel rating engine whose formulas reproduce the Python price for 100 of 100 holdout policies. | `artifacts/rating_engine_final_glm.xlsx`, `tests/test_rating_engine.py` |
| Corrected a Gini calculation that ranked policies by exposure, and audited the earlier version of the project. | `docs/AUDIT.md`, `tests/test_metrics.py` |
| Wrote a pytest suite of 60 tests. | `tests/` |

## 10. Do not claim

- "The GBM outperformed the GLM" or any percentage improvement from the GBM. On the final factors the difference is within noise. The nine-factor advantage depends on the suspect factor.
- "Closed 66% of the GLM-to-GBM gap with one term", or any share-of-gap figure, as a modelling improvement. That term relies on the suspect `BonusMalus` pattern.
- "SHAP-guided interactions improved the GLM". The selected interaction was not significant and is not in the final model.
- "Proved" or "discovered" leakage in freMTPL2. The evidence is strong but unproven, and the mechanism is not established.
- Gini 0.43 to 0.48, or any Gini improvement.
- Isotonic calibration.
- Dislocation analysis, Lorenz curves or a deployed dashboard, until the new app is merged and deployed.
- That the final model is suitable for real pricing. It omits bonus-malus, which a real insurer would use.
- The additive GBM's 84% / 16% split as a result about the final model. It was measured with `BonusMalus` included.

## 11. Tools and techniques actually used

- **Languages and libraries:** Python 3.13, pandas, NumPy, statsmodels, LightGBM, SHAP, scikit-learn (data download and `StratifiedGroupKFold`), matplotlib, Streamlit, Altair, openpyxl, pytest, Git. Also Excel formulas (`LOOKUP`, `INDEX`/`MATCH`, named ranges).
- **Techniques:**
  - Poisson and Gamma GLMs with log links, exposure offsets and prior weights;
  - large-loss capping with a flat load;
  - grouped, stratified cross-validation;
  - random-search hyperparameter tuning with early stopping;
  - SHAP values and SHAP interaction values;
  - paired bootstrap intervals;
  - Poisson deviance, Gini and Lorenz curves, actual against expected, double-lift charts;
  - IRLS, Newton-Raphson and Fisher scoring;
  - target-leakage diagnosis;
  - fixing a model specification before holdout scoring;
  - code audit and test-driven validation.

## 12. Links

- **Repository:** https://github.com/idk-what-to-put-here-32/Insurance-Pricing-Frequency-Severity-Modelling. This is the git remote. The earlier README gave `github.com/leo-yuu32/...`; check which is current.
- **App:** https://insurance-pricing-dashboard.streamlit.app/
- **Does the deployed app reflect the latest results?** No. All work since the audit is on local branches and has not been pushed. The live app is the original version, which fits the legacy models at start-up. After the branch is merged into `main`, Python 3.13 should be selected in the app's Advanced settings on Streamlit Community Cloud.
