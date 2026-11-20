# Motor claim frequency: GLM and GBM on freMTPL2

A frequency-severity pricing model for French motor third-party liability insurance (the freMTPL2 dataset), and a fair comparison of a Poisson GLM with a LightGBM model on the same data, rating factors, split and metrics.

Dashboard: https://insurance-pricing-dashboard.streamlit.app/

## Results

All figures are on a holdout of 133,726 policies (70,827 policy-years, 5,215 claims) that was not used for fitting, tuning or choosing any model term.

**Final models: eight rating factors, without `BonusMalus`.**

| Frequency model | Poisson deviance | Deviance explained | Gini | Actual / expected |
|---|---|---|---|---|
| Null (training frequency) | 0.47703 | 0 | 0 | |
| GLM (statsmodels, 55 parameters) | 0.46866 | 1.75% | 0.179 | 0.998 |
| GBM (LightGBM, tuned by cross-validation) | 0.46833 | 1.82% | 0.183 | 0.998 |

The GLM's deviance exceeds the GBM's by 0.00033, with a 95% paired bootstrap interval of -0.00019 to 0.00085. The interval includes zero: on these eight factors the GBM does not beat the GLM. The double-lift chart (`plots/double_lift_final_glm_vs_gbm.png`) shows the same thing. Where the two models disagree, the actual frequency lies between them.

**Comparison with `BonusMalus` (nine factors), set aside.**

| Frequency model | Poisson deviance | Deviance explained | Gini |
|---|---|---|---|
| GLM | 0.45391 | 4.85% | 0.297 |
| GLM + bonus-malus ladder indicator | 0.44991 | 5.69% | 0.315 |
| Additive GBM (no interactions) | 0.44876 | 5.93% | 0.319 |
| GBM | 0.44780 | 6.13% | 0.320 |

With `BonusMalus` the GBM leads by 0.0061 (interval 0.0049 to 0.0073), and one extra GLM term closes two thirds of that gap. That term flags bonus-malus scores that can only be reached through a claim. The result prompted a check of the factor itself, described below. These figures are kept as the analysis that exposed the problem, not as a pricing result.

### Caveats

- **`BonusMalus` timing.** The dataset does not say when the bonus-malus coefficient was recorded. Three checks on the training data suggest it was recorded after some of the claims it is used to predict:
  - its effect is about three times stronger in full-year records than in records under half a year;
  - in full-year records, 32% of policies whose score was raised at the last renewal have a claim in the same period, against 3.6% at the minimum score;
  - which scores carry the excess frequency shifts with the policy number.

  The evidence is strong but unproven, so the final models leave the factor out. A real insurer knows each driver's coefficient at the start of the policy and would use it; the final model is what this dataset supports cleanly. Details: `docs/DECISIONS.md`, D30 to D32.
- **Holdout reuse.** The holdout was used to compare several nine-factor models, and one selection decision (dropping an interaction term) was made with those results in view. The final models were specified and committed before they were scored, but the final figures may be mildly optimistic.
- **Mis-calibration by length of record.** Claim frequency is not proportional to exposure in this data: on the training data it is 0.054 claims per policy-year for full-year records and 0.112 for records under half a year. Both models take exposure as a log offset with its coefficient fixed at 1, which assumes proportionality. So on the holdout both over-predict full-year records (actual over expected 0.80) and under-predict records under half a year (1.43). The same pattern appears on the training data (0.77 and 1.42 for the final GLM). The standard fix is to fit log(exposure) as a covariate with its own coefficient instead of as a fixed offset; it was not applied because the final models had already been specified and scored on the holdout, and changing them in response to this result would be one more choice made with the holdout in view. Before scoring, exposure had also been kept out of the models deliberately (`docs/DECISIONS.md`, D3), because it is not known why short records claim more, and a covariate could learn how records are created rather than risk.

## Method

**Data.** freMTPL2freq and freMTPL2sev from OpenML (IDs 41214 and 41215). Policies with `IDpol` up to 24,951 are excluded: every one of them has a claim count, but almost none has a claim amount, so they are not a random sample of the book. Claim counts are taken from the severity records and capped at 4; exposure is capped at 1. Individual claims are capped at EUR 181,271, the 99.9th percentile of training claims, and the excess is reported as one flat large-loss load (1.192). 668,626 policies remain.

**Split.** Five folds with `StratifiedGroupKFold` (seed 42), stratified on whether a policy has a claim and grouped so that policies with identical rating factors stay together. One fold is the holdout; the other four are the training data and the cross-validation folds for every choice. The holdout policy IDs are in `artifacts/holdout_idpol.txt`.

**Models.**

- *Frequency GLM*: Poisson, log link, offset log(exposure), fitted by statsmodels without a penalty. Driver age (10 bands), vehicle age (5 bands), vehicle power, log(density), vehicle brand, fuel and region. Bands were chosen by cross-validated deviance on the training data. `Area` is left out because it is a banding of `Density`.
- *Frequency GBM*: LightGBM, Poisson objective, exposure as an offset through `init_score`, the same factors raw. Tuned by a 30-trial random search with 4-fold cross-validation and early stopping on the training data (learning rate 0.126, 17 leaves, at least 1,791 policies per leaf, 92 trees).
- *Severity*: one Gamma GLM (log link) on capped claims, shared by both frequency models. Only driver age improved cross-validated deviance, so it is the only factor.
- Each model gets one scalar rebalancing factor fitted on training data. All are within 0.3% of 1.

**Metrics.** Poisson deviance per policy-year on the holdout is the primary metric. Gini orders policies by predicted frequency and is twice the area between the Lorenz curve and the diagonal, so a model with no discrimination scores 0. Actual over expected is reported overall and by rating factor. Differences between models are given with a paired bootstrap interval (1,000 resamples). Pure premium metrics are secondary, because one claim of EUR 4.08 million dominates any loss-based metric.

**Independent checks.**

- `src/irls.py` is a GLM solver written from scratch in NumPy (iteratively reweighted least squares). On the final frequency model's design matrix (534,900 by 55) it reproduces the statsmodels coefficients to 6e-12 and the standard errors to 1e-8 (relative). For the Poisson log link each IRLS step is a Newton-Raphson step; for the Gamma log link it is Fisher scoring. Both statements are checked in `tests/test_irls.py`.
- `artifacts/rating_engine_final_glm.xlsx` prices a policy with Excel formulas from the final GLM's rating tables. It reproduces the Python price for 100 of 100 holdout policies.

The reasoning behind every choice is in `docs/DECISIONS.md`.

## Repository

```text
src/
  data.py                 loading, cleaning, the split
  design.py               bands and the GLM design matrix
  metrics.py              deviance, Gini, actual/expected, double lift, bootstrap
  glm.py, run_glm.py      statsmodels GLMs
  gbm.py, tune_gbm.py     LightGBM model and its tuning
  run_gbm.py              nine-factor GBMs
  run_interactions.py     SHAP analysis and GLM extensions (nine factors)
  check_bonus_malus_timing.py   the BonusMalus timing checks
  run_final.py            the final models
  summarise_models.py     results file read by the dashboard
  build_rating_engine.py  the Excel rating engine
  irls.py                 NumPy IRLS solver
  dashboard.py            Streamlit app
tests/                    pytest suite
outputs/                  results (JSON and model summaries)
plots/                    charts
artifacts/                holdout IDs and Excel workbooks
docs/                     DECISIONS.md, AUDIT.md, INTERVIEW_NOTES.md, CV_SUMMARY.md, REVIEW_PACK.md
legacy/                   the first version of the project, archived
```

## Reproducing the results

PowerShell, from the repository root. Times are compute times on a laptop.

```powershell
python -m pip install -r requirements-dev.txt
python src\run_glm.py                     # downloads the data and builds the split on first run (about 5 minutes)
python src\tune_gbm.py                    # nine-factor GBM tuning (about 35 minutes)
python src\tune_gbm.py additive           # additive GBM tuning (one to two hours)
python src\tune_gbm.py monotone           # constrained GBM tuning (about 75 minutes)
python src\run_gbm.py
python src\run_interactions.py            # about 20 minutes
python src\check_bonus_malus_timing.py    # about 6 minutes
python src\tune_gbm.py final              # final GBM tuning (about 10 minutes)
python src\run_final.py
python src\summarise_models.py
python src\build_rating_engine.py recalc  # "recalc" opens the workbook in Excel to compute and save its values
python -m pytest
```

## Dashboard

The app reads `outputs/model_comparison.json` only. It fits no model and does not import the modelling libraries, so it installs from `requirements.txt` alone.

```powershell
python -m pip install -r requirements.txt
streamlit run src\dashboard.py
```

On Streamlit Community Cloud, select Python 3.13 in the app's Advanced settings (the platform does not read `runtime.txt`).
