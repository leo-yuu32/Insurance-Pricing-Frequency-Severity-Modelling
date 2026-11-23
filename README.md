# Motor claim frequency: GLM and GBM on freMTPL2

This project builds a frequency-severity pricing model for French motor third-party liability insurance (the freMTPL2 dataset), and uses it to make a fair comparison of a Poisson GLM with a LightGBM model on the same data, rating factors, split and metrics.

Dashboard: https://insurance-pricing-dashboard.streamlit.app/

The rebuild of this project was developed with an AI coding assistant (Claude Code) under my direction.

The GBM's apparent advantage over the GLM comes from one factor. With all nine rating factors, the GBM's prediction error (Poisson deviance) on the holdout is 1.35% lower than the GLM's (95% interval 1.08% to 1.61%), and its SHAP values show that it relies most on the bonus-malus coefficient, a no-claims score. Checks on the training data suggest that this score was recorded after some of the claims it is used to predict, which is probable leakage, and so the final models leave it out. On the remaining eight factors the GBM's prediction error is 0.07% lower than the GLM's (95% interval -0.04% to 0.18%), which is small and not distinguishable from zero, and therefore on this data the GLM matches the GBM's accuracy while remaining fully transparent.

![The dashboard's header and summary](docs/screenshots/header.png)

![Without the no-claims score, the GBM's advantage all but disappears](docs/screenshots/article_without_the_score.png)

![Double-lift chart of the final GLM against the final GBM](docs/screenshots/double_lift.png)

## Aim

We ask how much a gradient-boosted model (GBM) improves on a Poisson GLM as a model of claim frequency. Pricing teams use GLMs because each rating factor has a table of relativities that can be read and checked, and so the comparison measures what that interpretability costs in accuracy.

## Results

In this section we give the results in the order in which we found them, beginning with a comparison on all nine rating factors and ending with the final comparison without `BonusMalus`. All figures are on a holdout of 133,726 policies (70,827 policy-years, 5,215 claims). The holdout was not used for fitting or tuning, and with one exception it was not used for choosing model terms either. The exception is a `BonusMalus` by `Region` interaction, which we dropped from the nine-factor GLM after seeing its holdout result (the bootstrap interval for its improvement included zero).

### First comparison: nine rating factors

We first compared the models on all nine freMTPL2 rating factors, including the bonus-malus coefficient `BonusMalus`.

| Frequency model | Poisson deviance | Deviance explained | Gini |
|---|---|---|---|
| GLM | 0.45391 | 4.85% | 0.297 |
| GLM + bonus-malus ladder indicator | 0.44991 | 5.69% | 0.315 |
| Additive GBM (no interactions) | 0.44876 | 5.93% | 0.319 |
| GBM | 0.44780 | 6.13% | 0.320 |

The GBM leads the GLM by 0.0061 (95% paired bootstrap interval 0.0049 to 0.0073). To find where this lead comes from, we computed SHAP values for the GBM on a sample of training policies. They point to `BonusMalus`, whose mean absolute SHAP value is three times that of the next factor, and its SHAP dependence plot is not a smooth curve but has spikes at the scores that can only be reached through a claim. We therefore gave the GLM one extra term flagging those scores (the ladder indicator), and this single parameter closes two thirds of the gap. The additive GBM, in which each tree uses one factor only, splits the gap approximately into the shapes of single factors (84%) and interactions between factors (16%), and so most of the GBM's lead comes from shapes that the GLM's bands only approximate.

### Investigation of `BonusMalus`

A single parameter closing two thirds of the gap looked too good, and so we investigated the factor itself, using the training data only. The dataset does not say when the bonus-malus coefficient was recorded. If it was recorded at the start of the policy, it is a legitimate rating factor. If it was recorded later, a claim in the data may already have raised it, and the factor then partly records the outcome it is used to predict.

Three checks suggest the second reading. The effect of `BonusMalus` is about three times stronger in full-year records than in records under half a year. In full-year records, 32% of policies whose score was raised at the last renewal have a claim in the same period, against 3.6% at the minimum score. And which scores carry the excess frequency shifts with the policy number. We therefore treat the factor as probable leakage. The evidence is strong but unproven, and so the final models leave the factor out. Note that a real insurer knows each driver's coefficient at the start of the policy and would use it, and so the final model is what this dataset supports cleanly. The checks are in `src/check_bonus_malus_timing.py`, and their results are in `outputs/bonus_malus_timing.json`.

### Final comparison: eight rating factors

We then fitted the final GLM and GBM on the eight remaining rating factors, with the GBM retuned by cross-validation on the training data, and scored each once on the holdout.

| Frequency model | Poisson deviance | Deviance explained | Gini | Actual / expected |
|---|---|---|---|---|
| Null (training frequency) | 0.47703 | 0 | 0 | |
| GLM (statsmodels, 55 parameters) | 0.46866 | 1.75% | 0.179 | 0.998 |
| GBM (LightGBM, tuned by cross-validation) | 0.46833 | 1.82% | 0.183 | 0.998 |

The GLM's deviance exceeds the GBM's by 0.00033, with a 95% paired bootstrap interval of -0.00019 to 0.00085, and so the GBM's edge is small and not distinguishable from zero. Furthermore, over 90% of the GBM's lead in the first comparison disappears once `BonusMalus` is removed. The double-lift chart (`plots/double_lift_final_glm_vs_gbm.png`) agrees with this. The GBM is closer to the actual frequency in six of the ten groups and the GLM in four, and in both end groups, where the two models disagree most, the actual frequency lies between them.

### Caveats

We note two further limitations of these results.

The first is holdout reuse. The holdout was used to compare several nine-factor models, and one selection decision (dropping an interaction term) was made with those results in view. The final models were specified and committed before they were scored (commit [`f954934`](https://github.com/leo-yuu32/Insurance-Pricing-Frequency-Severity-Modelling/commit/f954934)), but the final figures may still be mildly optimistic.

The second is mis-calibration by length of record. Claim frequency is not proportional to exposure in this data: on the training data it is 0.054 claims per policy-year for full-year records and 0.112 for records under half a year. Both models take exposure as a log offset with its coefficient fixed at 1, which assumes proportionality, and therefore on the holdout both over-predict full-year records (actual over expected 0.80) and under-predict records under half a year (1.43). The same pattern appears on the training data (0.77 and 1.42 for the final GLM). The standard fix is to fit log(exposure) as a covariate with its own coefficient instead of as a fixed offset. We did not apply it, since the final models had already been specified and scored on the holdout, and changing them in response to this result would be one more choice made with the holdout in view. Note also that, before scoring, we had deliberately kept exposure out of the models as a covariate. It is not known why short records claim more, and so a covariate could learn how records are created rather than risk.

## Method

In this section we describe the data, the split, the models and the metrics, in the order in which the pipeline builds them.

### Data and split

We use freMTPL2freq and freMTPL2sev from OpenML (IDs 41214 and 41215). We exclude policies with `IDpol` up to 24,951, since every one of them has a claim count but almost none has a claim amount, and so they are not a random sample of the book. Claim counts are taken from the severity records and capped at 4, and exposure is capped at 1. Individual claims are capped at EUR 181,271, the 99.9th percentile of training claims, and the excess is reported as one flat large-loss load (1.192). This leaves 668,626 policies.

We cut the policies into five folds with `StratifiedGroupKFold` (seed 42), stratified on whether a policy has a claim and grouped so that policies with identical rating factors stay together. One fold is the holdout, and the other four are the training data and the cross-validation folds for every choice. The holdout policy IDs are in `artifacts/holdout_idpol.txt`.

### Models

The frequency GLM is Poisson with a log link and offset log(exposure), fitted by statsmodels without a penalty. Its factors are driver age (10 bands), vehicle age (5 bands), vehicle power, log(density), vehicle brand, fuel and region, and the bands were chosen by cross-validated deviance on the training data. We leave out `Area`, because it is a banding of `Density`.

The frequency GBM is a LightGBM model with a Poisson objective, which takes exposure as an offset through `init_score` and the same factors raw. We tuned it by a 30-trial random search with 4-fold cross-validation and early stopping on the training data (learning rate 0.126, 17 leaves, at least 1,791 policies per leaf, 92 trees). Intuitively, the GLM can only use the shapes we give it through its bands and terms, whereas the GBM learns the shape of each factor, and how the factors combine, from the data.

For severity, both frequency models share one Gamma GLM (log link) on capped claims. Only driver age improved cross-validated deviance, and so it is the only factor. Each model gets one scalar rebalancing factor fitted on training data, and all are within 0.3% of 1.

### Metrics

Our primary metric is Poisson deviance per policy-year on the holdout. Gini orders policies by predicted frequency and is twice the area between the Lorenz curve and the diagonal, so that a model with no discrimination scores 0. We report actual over expected overall and by rating factor, and give differences between models with a paired bootstrap interval (1,000 resamples). Pure premium metrics are secondary, since one claim of EUR 4.08 million dominates any loss-based metric.

### Independent checks

We check the pipeline in two independent ways. First, `src/irls.py` is a GLM solver written from scratch in NumPy (iteratively reweighted least squares). On the final frequency model's design matrix (534,900 by 55) it reproduces the statsmodels coefficients to 6e-12 and the standard errors to 1e-8 (relative). For the Poisson log link each IRLS step is a Newton-Raphson step, whereas for the Gamma log link it is Fisher scoring, and both statements are checked in `tests/test_irls.py`. Second, `artifacts/rating_engine_final_glm.xlsx` prices a policy with Excel formulas from the final GLM's rating tables, and it reproduces the Python price for 100 of 100 holdout policies.

## Repository

The code is laid out as follows.

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
  save_final_gbm.py       the final GBM, refitted, checked and saved for the dashboard
  summarise_models.py     one results file for every model
  build_dashboard_data.py the small files the dashboard reads
  build_rating_engine.py  the Excel rating engine
  irls.py                 NumPy IRLS solver
  dashboard.py            Streamlit app
tests/                    pytest suite
outputs/                  results (JSON and model summaries)
plots/                    charts
artifacts/                holdout IDs, Excel workbooks and the dashboard's data files
legacy/                   the first version of the project, archived
```

## Reproducing the results

To reproduce the results, run the following in PowerShell from the repository root. The times given are compute times on a laptop.

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
python src\save_final_gbm.py              # refits the final GBM once, checks it against the saved predictions, saves it
python src\summarise_models.py
python src\build_dashboard_data.py        # the small files the dashboard reads (a few seconds)
python src\build_rating_engine.py recalc  # "recalc" opens the workbook in Excel to compute and save its values
python -m pytest
```

## Dashboard

The app reads the small precomputed files in `artifacts/dashboard/`, which `src/build_dashboard_data.py` writes from the saved results, and for its pricing tool it loads the saved final GBM (`artifacts/gbm_final.txt`) with LightGBM for prediction only. Since it fits no model, it installs from `requirements.txt` alone.

```powershell
python -m pip install -r requirements.txt
streamlit run src\dashboard.py
```

On Streamlit Community Cloud, select Python 3.13 in the app's Advanced settings, since the platform does not read `runtime.txt`.
