# Phase 0 audit: French MTPL pricing engine

Date: 11 November 2026. Scope: read-only audit of the repo as it stands on `main` (commit `3368afe` plus three uncommitted files), before any GBM work.

Nothing in `src/`, `artifacts/`, `plots/`, `README.md` or `CLAUDE.md` was changed. No problem described here has been fixed.

**How the numbers were produced.** Every figure below comes from running the existing pipeline (`run_pricing_pipeline`) unmodified and analysing its output in separate scripts kept outside the repo. The run reproduces `artifacts/scalars.json` exactly (off-balance factor 0.8156706832, Gini 0.4265437160 and 0.4811506233, average premium 115.6459655339), so the figures describe the same pipeline the dashboard runs. Environment: Python 3.13.9, numpy 2.3.4, pandas 2.3.3, scikit-learn 1.8.0, statsmodels 0.15.0. Where a statement comes from reading code and was not measured, it says so.

---

## 1. Summary

Five findings matter most. The first three affect lines on the CV.

1. **The Gini function ranks policies by the wrong quantity.** It sorts by premium divided by exposure, but the premium is already an annual rate. Under this function a constant prediction (every policy priced the same) scores **0.444**, which is higher than the model's 0.427. The reported Gini measures an exposure pattern in the data, not the model.
2. **The move from 0.43 to 0.48 was not a model change.** It is the difference between two price columns on the same test set. The second column is the first after isotonic calibration and seven hand-set multipliers, which git history shows were chosen from charts drawn on that test set. With the ordering corrected, the same comparison is 0.075 to 0.361, but almost all of that comes from a single claim of EUR 4.08 million on a 19-year-old in the test set. Excluding that one policy it is 0.042 to 0.109.
3. **More than a quarter of the claims have no claim amount.** 9,116 policies (26.8% of policies with a claim) have `ClaimNb > 0` and no severity record. They count as claims in the frequency model and as zero loss everywhere else. The off-balance factor of 0.816 scales this away without anyone having to notice it.
4. **The final price is far below the loss cost.** Actual losses on the test set are 1.73 times the final price. The isotonic step calibrates to losses capped at EUR 15,000 per policy, which removes 28.6% of training losses, and nothing puts that back.
5. **The GLMs are scikit-learn with a ridge penalty, not statsmodels, and they use seven factors.** The penalty shrinks the driver-age effect so far that ages 18 to 21 are priced at the portfolio average frequency (0.101 predicted against 0.229 observed, on the training data). `BonusMalus` and `Density` are not in the model at all.

**What this means for the CV.** Of the five current claims, two are supported, two need rewording and the Gini claim is unsupported (section 11). The Gini line should come off now, not after the rebuild.

---

## 2. Repo structure and data flow

```text
src/config.py                 constants: split, penalty, caps, bands, feature list, loadings
src/data_preprocessing.py     fetch from OpenML, merge severity onto policies, filter, band
src/frequency_model.py        scikit-learn PoissonRegressor in a one-hot pipeline
src/severity_model.py         scikit-learn GammaRegressor in a one-hot pipeline
src/pricing_engine.py         split, fit, off-balance, isotonic, loadings, Gini, all charts
src/dashboard.py              Streamlit app; calls run_pricing_pipeline() at start-up
src/export_model_artifacts.py writes coefficients, knots, loadings, 100 sample policies
src/build_excel_engine.py     builds artifacts/rating_engine.xlsx from those files
artifacts/                    scalars.json and rating_engine.xlsx are tracked; the CSVs are git-ignored
plots/                        six PNGs, tracked
```

There are no tests, no `docs/` folder before this file, and no notebook.

Data flow, in order:

1. `fetch_raw_data` downloads `freMTPL2freq` (OpenML 41214) and `freMTPL2sev` (41215).
2. `preprocess_data` sums claim amounts per policy, left-joins them onto the frequency file, computes `AvgSeverity = TotalLoss / ClaimNb`, then filters.
3. `bin_features` bands driver age, vehicle age, vehicle power and brand.
4. `train_test_split(test_size=0.2, random_state=42)` splits 80/20.
5. Frequency and severity GLMs are fitted on the training set.
6. Raw pure premium = predicted frequency times predicted severity.
7. An off-balance factor (training actual losses over training predicted losses) scales it: `PurePremium_OB`.
8. Isotonic regression, fitted on the training set, maps it to `PurePremium_Calibrated`.
9. Seven multipliers from `config.ACTUARIAL_LOADINGS` give `Final_Price`.
10. Gini, lift charts, one-way charts, a Lorenz curve and a dislocation histogram are all computed on the test set.

The dashboard runs steps 1 to 8 once at start-up (cached), then reruns step 9 and the metrics every time a slider moves. The Excel workbook replicates steps 6 to 9 for 100 test policies from exported coefficients.

`main()` and `run_pricing_pipeline()` in [src/pricing_engine.py](../src/pricing_engine.py) contain the same pipeline written out twice.

---

## 3. How the data is loaded and cleaned

### Raw data

| | Count |
|---|---|
| Policies in `freMTPL2freq` | 678,013 |
| Claims in `freMTPL2freq` (sum of `ClaimNb`) | 36,102 on 34,060 policies |
| Rows in `freMTPL2sev` | 26,639 on 24,950 policies |
| Total claim amount in `freMTPL2sev` | EUR 60,697,931 |

### Filters ([src/data_preprocessing.py:44-47](../src/data_preprocessing.py#L44-L47))

| Step | Effect |
|---|---|
| Drop `Exposure <= 0.003` | 3,105 policies removed, carrying 50 claims |
| Clip `Exposure` at 1.0 | 1,224 policies clipped (maximum was 2.01) |
| Drop `ClaimNb >= 10` | 4 policies removed (claim counts 11, 11, 11 and 16) |
| Rows remaining | 674,904 |

Two points an actuary would raise:

- The smallest exposure in the data is 0.0027, which is one day. The first filter therefore removes exactly the one-day policies. The code comment calls them "likely data errors" with no evidence. They carry 50 claims on about 8 policy-years, so dropping them is not neutral.
- Claim counts are dropped at 10 or more, not capped. Four rows is immaterial, but the choice is not documented.

### Joining claims to policies

The severity file is summed per policy and left-joined on `IDpol`. `ClaimNb` is taken from the frequency file and never compared with the number of severity rows.

| Check | Result |
|---|---|
| Policies with `ClaimNb > 0` and **no** severity record | **9,116** (26.76% of claim policies), carrying 9,657 claims (26.75% of all claims) |
| Policies with `ClaimNb = 0` but severity rows | 0 |
| Matched claim policies where severity rows equal `ClaimNb` | 24,943 of 24,944 (one policy has one row fewer) |
| Severity `IDpol` values absent from the frequency file | 6 policies, 195 rows, EUR 788,714 (1.3% of all claim amount), silently dropped by the left join |
| Zero-amount severity rows | 0 |

The unmatched claims are not scattered. They sit in one block at the start of the file:

- All 9,139 policies with `IDpol <= 24,500` have at least one claim, and 8,851 of them have no severity record.
- Above `IDpol` 30,000 only 21 of 24,546 claim policies are unmatched.

So this is a block of policies whose claim amounts are missing from the severity file, not a set of nil claims spread through the book. I believe published treatments of this dataset reconcile the counts to the severity file, but that is from memory and should be checked when the convention is chosen in Phase 1.

**Consequences in the current pipeline:**

- The frequency model counts these 9,657 claims.
- The severity model excludes them, because it filters on `AvgSeverity > 0` (19,884 of the 27,235 claim policies in the training set are used).
- Frequency times severity therefore overstates losses by roughly a third.
- The off-balance factor hides it. As rough arithmetic: about 27% of claims have no amount (factor 0.73), and the severity model is fitted to losses with 9.9% capped away (divide by 0.90), which gives about 0.81. The actual factor is 0.816. An off-balance factor this far from 1 is a symptom, and here it is two errors partly cancelling.

### Train and test sets after cleaning

| | Rows | Exposure | Claims | Frequency | Total loss | Loss cost |
|---|---|---|---|---|---|---|
| Train | 539,923 | 286,923.5 | 28,854 | 0.1006 | 45,522,099 | 158.66 |
| Test | 134,981 | 71,426.7 | 7,149 | 0.1001 | 14,304,768 | 200.27 |

The test loss cost is 26% above the training loss cost because of one claim (section 8).

---

## 4. Train/holdout split and what was tuned on the holdout

**The split.** One fixed split exists: `train_test_split(df, test_size=0.2, random_state=42)` at [src/pricing_engine.py:611](../src/pricing_engine.py#L611). It is not stratified. It is made after filtering and banding, and it is deterministic as long as the data and the filters do not change. There is no validation set and no cross-validation anywhere.

**What was set using the test set:**

1. **The seven loadings.** Commit `86123be` shows `apply_loadings` being written with comments such as "Cost ~130 vs Price ~116. Factor = 1.12" and "Cost ~50 vs Price ~90. Factor = 0.55". The only one-way charts in that version are drawn on `test_results`. The multipliers were read off test-set charts and then scored on the same test set. The current `main()` still prints a "VERIFICATION: DriverAge 18-21 Bin" block computed on the test set.
2. **The ridge penalty**, probably. `alpha` changed from 0.001 to 0.01 between commits `b3539a4` and `593bd8a` with no recorded reason. At that time the script's only outputs were test-set charts. This is inferred, not proven.

**Smaller point.** The top-five brand list in `bin_features` is computed on the full data before the split. It uses feature counts only, so the leak is negligible.

**Conclusion.** The existing test set has been used for selection and cannot serve as an untouched holdout for the rebuilt project. A new split is needed.

---

## 5. Exactly what produced the Gini move from 0.43 to 0.48

**In one sentence:** 0.43 is the Gini of the off-balanced pure premium and 0.48 is the Gini of the final price, both on the same test set in the same run, so the "improvement" is the effect of isotonic calibration plus seven hand-set multipliers, measured by a function that ranks policies incorrectly.

It was not an age re-banding. The driver-age bands have not changed since the first version of the code.

### Step by step, under the project's own Gini function

| Premium | Gini | Change |
|---|---|---|
| Constant prediction (no model) | 0.4443 | |
| Off-balanced pure premium (**the CV's 0.43**) | 0.4265 | |
| After isotonic calibration | 0.4386 | +0.0121 |
| Plus the 18 to 21 loading (x2.40) only | 0.4620 | +0.0234 |
| Plus the 22 to 25 loading (x1.40) | 0.4624 | +0.0004 |
| Plus the other five loadings (**the CV's 0.48**) | 0.4812 | +0.0188 |

So the 18 to 21 multiplier accounts for 0.023 of the 0.055. The isotonic step contributes 0.012 only because it collapses the premium to 37 distinct values on the test set, and the function then breaks the ties by exposure.

### In model terms

- The ridge penalty (`alpha = 0.01`) shrinks every driver-age coefficient towards the reference level, which happens to be 18 to 21. The fitted relativities for all eight bands lie between 0.89 and 1.12.
- The model therefore predicts a frequency of 0.101 for ages 18 to 21 on the training data. The observed frequency is 0.229. Actual over expected is 2.28.
- This is caused by the penalty, and it can be said without refitting: an unpenalised Poisson GLM with a log link reproduces the observed claim total for every level of every categorical factor on its training data. An actual-over-expected of 2.28 on the training data for a level that is in the model can only come from the penalty.
- The x2.40 multiplier is a manual patch for that shrinkage, applied after the model, to a band holding 1.2% of exposure.

### Is it a level correction or a ranking change?

It is a relativity change to one segment, so it does change the ranking, which is why Gini moves. But the size of the move is not evidence of better ranking, for two reasons.

**Reason 1: one claim.** The largest claim in the whole dataset, EUR 4,075,401, is in the test set, on a 19-year-old with exposure 0.22. It is 28.5% of all test-set losses. Any change that moves the 18 to 21 band up the ranking captures it.

| Premium | Existing function, all policies | Existing function, excluding that one policy |
|---|---|---|
| Constant prediction | 0.4443 | 0.2576 |
| Off-balanced (the 0.43) | 0.4265 | 0.2482 |
| Isotonic calibrated | 0.4386 | 0.2587 |
| Plus the 18 to 21 loading only | 0.4620 | 0.2583 |
| Final price (the 0.48) | 0.4812 | 0.2809 |

With that single policy removed, the 18 to 21 loading adds nothing (0.2587 to 0.2583).

**Reason 2: the loadings make the ranking of claim counts worse.** Ordering test policies by each premium and measuring how well that ranks claim counts (corrected ordering):

| Premium | Gini on claim counts |
|---|---|
| Off-balanced | 0.1286 |
| Final price | 0.0852 |

I added the two checks in this subsection after seeing the first results, because a jump this large from one multiplier looked too good and the working rules say to suspect the result before reporting it.

---

## 6. Isotonic calibration

**Where it sits.** After the off-balance factor and before the loadings ([src/pricing_engine.py:630-643](../src/pricing_engine.py#L630-L643)).

**What it is fitted on.** The training set: off-balanced premium as input, `min(TotalLoss, 15000) / Exposure` as the target, weighted by exposure ([src/pricing_engine.py:34-53](../src/pricing_engine.py#L34-L53)). The inputs are in-sample predictions from GLMs fitted on the same rows.

**What it is evaluated on.** The test set. It is not fitted on data it is later evaluated on, so there is no direct leak.

**What is wrong with it:**

- **It changes the level, not only the shape.** The target is capped at EUR 15,000 per policy, which removes 28.6% of training losses. The calibrated premium is therefore on a capped basis, and no loading restores the excess.
- **It produces a step function.** 61 knots, 31 distinct output levels, 37 distinct premiums on the test set against 42,419 before calibration.
- **It prices some policies at zero.** The lowest step is exactly 0. Seven test policies (0.01%) get a premium of zero. The count is small, but a rating structure that can output zero is a defect.
- **It cannot improve ranking.** Isotonic regression is monotone, so it preserves order except where it creates ties.

**Actual versus expected on the test set, by stage:**

| Stage | A/E against actual losses | A/E against losses capped at 15,000 |
|---|---|---|
| Raw (frequency times severity) | 1.031 | 0.599 |
| Off-balanced | 1.264 | 0.734 |
| Isotonic calibrated | 1.769 | 1.027 |
| Final price | 1.732 | 1.006 |

Two things follow.

- The final price recovers 58% of actual test-set losses. Even allowing for the one very large claim in the test set, the training-basis shortfall is 28.6%.
- The lift and one-way charts plot actual losses capped at EUR 15,000. Against that basis the "technical" premium looks 36% too high (A/E 0.734) and the calibrated one looks right (1.027). The dashboard's message that the commercial view tracks actuals better is produced by comparing each premium with the capped basis that only one of them was fitted to.

**By driver-age band (test set, frequency):**

| Band | Exposure | Actual frequency | Predicted frequency | A/E |
|---|---|---|---|---|
| 18 to 21 | 868 | 0.213 | 0.100 | 2.13 |
| 22 to 25 | 2,368 | 0.171 | 0.118 | 1.45 |
| 26 to 30 | 5,809 | 0.104 | 0.106 | 0.98 |
| 31 to 40 | 16,943 | 0.091 | 0.094 | 0.97 |
| 41 to 50 | 17,583 | 0.101 | 0.105 | 0.97 |
| 51 to 60 | 15,039 | 0.093 | 0.099 | 0.94 |
| 61 to 75 | 10,199 | 0.095 | 0.098 | 0.96 |
| 75+ | 2,618 | 0.102 | 0.102 | 1.01 |

The same pattern holds on the training set (2.28 and 1.29 for the two youngest bands).

---

## 7. The 99.9th-percentile large-loss cap

- **Computed on training data only.** The cap is taken inside `train_severity_model`, after the split ([src/severity_model.py:22-23](../src/severity_model.py#L22-L23)). There is no leak. Training value EUR 129,390; the full-data value would be EUR 133,329.
- **It is a cap on the policy-level average severity, not on individual claims.** The 99.9th percentile of individual claim amounts is EUR 162,784.
- **Effect.** 20 training rows are capped, removing EUR 4.50 million, which is 9.9% of training losses.
- **The excess is restored only by accident.** The off-balance factor balances to uncapped training losses, which spreads the capped amount back proportionally. The isotonic step then removes far more.
- **The cap that actually drives the price is the other one.** `CALIBRATION_CAP_THRESHOLD = 15000` caps each policy's total loss inside the calibrator. It affects 261 training policies and removes 28.6% of training losses (41.9% on the test set). The CV mentions only the 99.9th percentile.

---

## 8. How Gini is computed

`calculate_gini` at [src/pricing_engine.py:412-442](../src/pricing_engine.py#L412-L442):

1. Computes `pred_rate = pred / Exposure`.
2. Sorts ascending by `pred_rate`.
3. Cumulates exposure (x axis) and actual `TotalLoss`, uncapped (y axis), each normalised to 1.
4. Returns `1 - 2 * area under the curve` by the trapezium rule.

Steps 2 to 4 are the standard exposure-weighted Lorenz construction. **Step 1 is the bug.** The premiums passed in are already annual rates, so dividing by exposure again ranks policies by `rate / Exposure`. On the test set predicted rates vary by a factor of 7 and exposure by a factor of 183. The ordering is therefore dominated by `1 / Exposure`.

That matters because short-exposure policies in this dataset have very high observed claim rates:

| Exposure band (test) | Policies | Claim frequency |
|---|---|---|
| up to 0.05 | 9,452 | 1.038 |
| 0.05 to 0.10 | 15,611 | 0.304 |
| 0.10 to 0.25 | 19,061 | 0.208 |
| 0.25 to 0.50 | 26,375 | 0.126 |
| 0.50 to 0.75 | 18,525 | 0.098 |
| 0.75 to 1.00 | 12,185 | 0.087 |
| exactly 1.00 | 33,772 | 0.071 |

Sorting by `1 / Exposure` alone therefore "ranks risk" well, with no model. That is the 0.444 for a constant prediction.

`plot_lorenz_curve` contains a copy of the same code with the same bug. `plot_lift_curve` sorts by the premium directly, which is correct, so the lift chart and the Gini on the same dashboard page disagree about the ordering.

### Corrected figures

Sorting by the predicted rate, pooling ties, same axes:

| Premium | All losses | Excluding the largest policy | Losses capped at 15,000 |
|---|---|---|---|
| Constant prediction | 0.000 | 0.000 | 0.000 |
| Off-balanced | 0.075 | 0.042 | 0.057 |
| Isotonic calibrated | 0.079 | 0.049 | 0.061 |
| Final price | 0.361 | 0.109 | 0.116 |

Two conclusions:

- The current model's ranking power on pure premium is small: about 0.04 to 0.08 depending on how one claim is treated.
- A pure premium Gini on this test set is not a stable number. One policy is 28.5% of the losses and the ten largest are 38.6%.

### Baseline frequency metrics of the current model (test set)

These are the "before" figures for Phase 1. They use `ClaimNb` as it stands, including the claims with no amount, so they will not be like for like once the data conventions change.

| Metric | Model | Null model (training mean frequency) |
|---|---|---|
| Poisson deviance per unit exposure | 0.6123 | 0.6201 |
| Poisson deviance per policy | 0.3240 | 0.3281 |
| Deviance explained | 1.25% | |
| Frequency Gini (corrected ordering, actual = claim counts) | 0.148 | 0 |

---

## 9. The NumPy GLM

**It is not in this repo.** There are three candidates in sibling folders.

### `Projects/GLM From Scratch/numpy-glm-from-scratch/poisson_glm_from_scratch.ipynb`

This is the one the CV describes.

| Aspect | What it does |
|---|---|
| Form | A notebook function, `train_poisson_glm(X, y, exposure, max_iter=10)`; not an importable module |
| Family and link | Poisson, log link only |
| Offset | Yes. `mu = exposure * exp(X beta)`, and the working response `z = eta + (y - mu) / mu` correctly leaves the offset out |
| Prior weights | No |
| Gamma | No |
| Convergence | No test. Fixed 10 iterations from `beta = 0` |
| Outputs | Coefficients only. No deviance, standard errors or dispersion |
| Design matrix | `[1, DrivAge, VehPower]`, both as raw linear terms. No banding, no dummy variables |
| Data | Exposure clipped at 1, no other filter; its own 80/20 split by `np.random.permutation`, seed 42, different from this repo's split |

**Check against statsmodels.** Running the same update on the notebook's own data and split, the coefficients agree with statsmodels to 2e-13 from iteration 8 onwards. Ten iterations are enough for this three-parameter model. The algorithm is correct.

Its README has two inaccuracies. It writes the update as `beta_new = beta_old + (X'WX)^-1 X'Wz`, which mixes two forms (the code correctly uses `beta_new = (X'WX)^-1 X'Wz`). It also says convergence is tracked through the mean squared error, which the code does not do.

**Gaps against a validated reimplementation of the pipeline GLMs:** move to a module, add prior weights, the Gamma family with log link, a convergence criterion, deviance, standard errors and the Pearson dispersion estimate, and accept the pipeline's design matrix.

### `Projects/GLM and Pricing Analysis/.../src/glm_pipeline.py`

A `PoissonGLM` class with IRLS, a tolerance and a deviance history. It should not be used as a base. From reading the code, not from running it:

- It runs on synthetic data, not freMTPL2.
- The working response includes the offset (`z = eta + (y - mu) / mu` with the offset inside `eta`) and is then regressed on `X` without subtracting it, so it does not converge to the maximum likelihood estimate when exposure varies.
- It prints "All coefficients match (validation successful)!" without comparing anything.
- It builds an n-by-n diagonal weight matrix, which cannot work on 540,000 rows.

### `Projects/GLM/GLM.py`

An early Newton-Raphson exercise with row-by-row loops and no offset. Not relevant.

---

## 10. Issues ranked by severity

### Critical: would not survive an actuary's first questions

| # | Issue | Evidence |
|---|---|---|
| C1 | Gini ranks by premium / exposure. A constant prediction scores 0.444, above the model's 0.427 | Sections 5 and 8; [src/pricing_engine.py:421](../src/pricing_engine.py#L421) |
| C2 | The reported Gini improvement is seven hand-set multipliers chosen on the test set, and it depends on one claim | Section 5; commit `86123be` |
| C3 | 9,116 policies (26.8% of claim policies) have claims but no amount, and the pipeline does not reconcile them | Section 3 |
| C4 | The final price recovers 58% of test losses; calibration targets losses capped at EUR 15,000 and nothing restores the 28.6% removed | Section 6 |

### Major

| # | Issue | Evidence |
|---|---|---|
| M1 | Undocumented ridge penalty prices ages 18 to 21 at the average frequency (training A/E 2.28). The frequency model explains 1.25% of test deviance | Section 5; [src/config.py:28](../src/config.py#L28) |
| M2 | `BonusMalus` and `Density` are not in the model. `B12`, the third-largest brand by exposure (51,836 policy-years in training), is forced into "Other" | [src/config.py:87-95](../src/config.py#L87-L95), [src/data_preprocessing.py:85-89](../src/data_preprocessing.py#L85-L89) |
| M3 | One claim is 28.5% of test losses. No pure premium metric on this split is stable, and no uncertainty is reported anywhere | Section 8 |
| M4 | Isotonic step: 31 levels, a zero premium, and charts that compare capped actuals with an uncapped-basis premium | Section 6 |
| M5 | Dashboard text states things that are wrong (below) | [src/dashboard.py](../src/dashboard.py) |
| M6 | The existing test set was used for selection, so there is no clean holdout | Section 4 |
| M7 | No tests. Dependencies are mostly unpinned. `requirements.txt` says `numpy<2.0.0` while the local environment runs 2.3.4 | `requirements.txt` |

Dashboard statements for M5:

- "Notice how the Commercial Gini is higher than the Technical Gini, this is because we are applying loadings to the model to make it better." The loadings were tuned on the data they are scored on.
- "If the model is perfect, the curve will be a straight line at 45 degrees." The 45-degree line is a model with no discrimination.
- "The Gini Coefficient is the area between the blue curve and the diagonal line." It is twice that area.
- "Since most of our graph is green, this means we are applying loadings to the model to also make it better for the commercial market." Green bars are discounts; this does not follow.
- The charts are single-model lift charts. None is a double-lift chart, which compares two models.

### Minor

- One-day policies (3,105, with 50 claims) are dropped as "likely data errors" without evidence.
- The 99.9th percentile cap is applied to policy-average severity, not to individual claims.
- Six severity policies (195 rows, EUR 0.79 million) are dropped silently by the join.
- The pipeline is duplicated in `main()` and `run_pricing_pipeline()`.
- The dashboard fits models at start-up, and `plot_lorenz_curve` and `plot_dislocation_histogram` save a PNG into `plots/` on every render, overwriting tracked files.
- Loadings are applied with a row-wise `DataFrame.apply` over 135,000 rows on every slider move.
- The uncommitted edit to `src/dashboard.py` swaps the colour description to "predicted in blue, actual in green". The code plots predicted in green and actual in blue, so the edit makes the text wrong. The committed version was right.
- The GitHub link in the dashboard (`idk-what-to-put-here-32`) differs from the clone URL in the README (`leo-yuu32`).
- The CSV artifacts are git-ignored, so `build_excel_engine.py` cannot run from a fresh clone.
- `rating_engine.xlsx` contains two sheets, `Sample_Policy` and `Sensitivity_OneWay`, that the build script does not create. They were added by hand, and rerunning the script would overwrite the file and lose them.
- `VehGas` levels carry literal quote marks from OpenML (`'Regular'`).
- `plotly` is in `requirements.txt` and is not used.
- The top-five brand list is computed before the split.

---

## 11. Status of the five CV claims

| # | Claim | Status | What is accurate |
|---|---|---|---|
| 1 | "Built a frequency-severity pricing model (Poisson and Gamma GLMs) on the French Motor Third-Party Liability dataset" | **Supported** | True as worded. The GLMs are scikit-learn with a ridge penalty, on seven banded factors |
| 2 | "Capped large losses at the 99.9th percentile and applied isotonic calibration to align predicted and observed loss cost" | **Needs rewording** | The cap is real and is computed on training data only. The isotonic step aligns the price with losses capped at EUR 15,000 per policy, not with observed loss cost: against actual losses the final price has A/E 1.73 on the test set. The rebuild removes the isotonic step |
| 3 | "Improved the Gini coefficient from 0.43 to 0.48 by [cause]" | **Unsupported** | Section 5. Not a model change; measured by a function under which no model at all scores 0.44; the gain depends on one claim and on multipliers chosen on the test set |
| 4 | "Deployed a Streamlit dashboard for dislocation analysis, using double-lift charts and Lorenz curves ...; replicated core calculations in Excel to cross-validate the Python outputs" | **Needs rewording** | The dashboard, dislocation histogram and Lorenz curve exist. The charts are lift charts, not double-lift charts. The Excel workbook does reproduce the Python final price: 100 of 100 sample policies match, with a maximum difference of zero in the cached values. It replicates scoring from exported coefficients, not model fitting. I did not check the live deployment |
| 5 | "Implemented a Poisson GLM with log link and exposure offset in NumPy, fitted by IRLS (equivalent to Newton-Raphson under the canonical link)" | **Supported** | Matches statsmodels to 2e-13 on its own three-parameter model. The equivalence statement is correct. It has no convergence test and lives in a separate repo |

---

## 12. Critical review of the plan

The test applied throughout: would this hold up in front of a pricing actuary interviewing a graduate?

### What CLAUDE.md assumed that the code contradicts

| CLAUDE.md says | The code shows |
|---|---|
| GLMs fitted with statsmodels | scikit-learn `PoissonRegressor` and `GammaRegressor` with `alpha = 0.01` |
| Gini improved by a fix to the 18 to 21 segment | Two price columns compared on one test set; the "fix" is a x2.40 multiplier outside the model |
| Keep the existing Gini definition unchanged | The existing definition is wrong |
| Double-lift charts exist | Only single-model lift charts exist |
| The NumPy GLM may be in this repo | It is a notebook in another folder |

### Decisions already taken

- **Leo:** statsmodels, unpenalised, fits the GLMs in the pipeline. The NumPy IRLS solver is a from-scratch reimplementation validated against statsmodels in pytest, not the pipeline's engine.
- **Agreed after the second opinion:** remove isotonic calibration; remove the default loadings; add an additive GBM; fix the Gini ordering; settle data conventions first; frequency metrics primary; phase branches stay local.

### Phase by phase

**Phase 1 (shared foundations): sound in outline, wrong on Gini, and missing the data decisions.**

- "Gini, using the existing project definition unchanged" must go (C1).
- The claim-count convention has to be decided before anything is fitted (C3). Options, with my recommendation first:
  1. Define the claim count as the number of severity records per policy. Frequency and severity then describe the same claims, and frequency times severity reproduces total losses. The 9,116 policies become claim-free.
  2. Drop the block of policies with `IDpol` up to about 24,500 (1.3% of policies), on the grounds that every one of them has a claim and almost none has an amount, so the block is not a random sample of the book.
  3. Keep `ClaimNb` as it is for frequency and accept that pure premium needs a large correction. I would not do this.
- The existing test set cannot be reused (M6).
- The 0.43 and 0.48 figures should be retired, not re-reported on a new split.
- The check for duplicate risk profiles is cheap and can stay, but it is low priority.

**Phase 2 (from-scratch GLM): the solver swap no longer applies.**

- "Downstream results must be unchanged" was never achievable against a penalised fit, and Leo's decision removes the swap anyway.
- The valuable part remains and is cheap: tests of the NumPy solver against statsmodels on the pipeline's real training design matrix, for both models.
- Keep Gamma in the NumPy solver. With a log link the Gamma working weights are just the prior weights, which makes the Fisher scoring point concrete.
- This phase no longer blocks Phase 3. With interviews in January, the order 1, 3, 2, 4, 5 is safe if time is short.

**Phase 3 (LightGBM): sound, with one fairness problem and one missing measurement.**

- Both `init_score` traps are stated correctly.
- As written, the GBM would get raw factors that the GLM does not have. The gap would then mostly measure the missing `BonusMalus` (M2). Every model must use the same nine factors.
- No uncertainty is attached to the GLM-to-GBM gap. On this dataset the gap in Poisson deviance will be small, and the Phase 4 ratio divides by it. A paired bootstrap over holdout policies is cheap and answers "is that difference real?".
- "Applying any calibration step identically to both" should become one scalar rebalance per model, fitted on training data, with its value reported. In-sample isotonic on GBM training predictions would be overfitted.
- The additive GBM (each tree restricted to one feature) splits the gap into non-linear shape and interactions. It needs the same tuning protocol and should be described as a diagnostic, and the split as approximate.
- Exposure must stay an offset only and never be a feature. The table in section 8 shows how strongly exposure predicts claims in this data.

**Phase 4 (SHAP and interactions): sound, with a sharper headline.**

- Selecting interactions on training data only is right.
- Interactions can only close the interaction part of the gap. Report two ratios: the share of the total gap closed, and the share of the additive-GBM-to-GBM gap closed. Skip any ratio whose denominator is within noise.
- With about 500,000 training rows nearly any interaction is significant under a likelihood-ratio test, so that test is weak evidence on its own. The selection criterion should be cross-validated Poisson deviance on the training data, with the likelihood-ratio test reported as supporting evidence only.
- If the additive GBM matches the full GBM within noise, that is the finding. Do not force interactions into the GLM to produce a number.

**Phase 5 (dashboard and write-up): sound, but "every existing tab still works" is the wrong goal.**

- The existing tabs present the flawed Gini and the "loadings improve the model" message (M5). They need rebuilding on the new pipeline, not preserving.
- A multiplier estimated from loss data is a model term and belongs in the GLM with a standard error. A commercial loading is a business input. The sliders can stay as a scenario tool with defaults of 1.00, clearly labelled, with no "Commercial Gini".
- Dislocation is better shown between models: the premium change per policy from moving GLM to GBM. That is what a pricing team means by the term, and it keeps the dashboard line on the CV true.
- The unpenalised statsmodels fit on about 540,000 rows will probably not fit in Streamlit's free-tier memory, and the app currently fits at start-up. Precomputed outputs are required, not optional, and nothing should be pushed to `main` until the app reads them.
- Regenerate the Excel workbook once, for the final GLM. Write to a new file name so the two hand-built sheets are not lost.

### Effort against value

| Item | Effort | Value | View |
|---|---|---|---|
| NumPy solver tested on the real design matrix | Low | High | Do it |
| Additive GBM | Low | High | Do it |
| Paired bootstrap on the deviance gap | Low | High | Do it |
| Keeping Excel in step with every model | Medium | Low | Once, final GLM only |
| SHAP dependence plots in the dashboard | Medium | Medium | Keep to two or three factors |
| Duplicate-profile check | Low | Low | Keep, report in one line |
| Keeping the isotonic and loadings pipeline alive | Medium | Negative | Move to `legacy/` |

### Rating factors for the rebuilt GLM

freMTPL2 has nine rating factors. The GBM takes all nine raw.

| Factor | GLM treatment | Evidence and notes |
|---|---|---|
| `DrivAge` | Banded, the existing eight bands | Reference level moves from 18 to 21 (1.2% of exposure) to the largest band |
| `VehAge` | Banded, about five bands, age 0 separate from age 1 | Cut-points set on training data only |
| `VehPower` | Categorical, sparse high levels merged | Training exposure: 9 is 12,226; 10 is 12,206; 11 is 6,824; 12 to 15 are each under 3,100 |
| `BonusMalus` | Continuous, linear on the log scale, capped near 150 | 56.8% of policies are at 50; 1.2% are above 100; 0.03% are above 150. Use bands only if the training one-way shows a kink at 100 |
| `Density` | Continuous, as log(`Density`) | |
| `Area` | Not in the GLM if `Density` is | `Area` is a deterministic banding of `Density` (A is 1 to 50, B 50 to 100, C 100 to 500, D 500 to 2,000, E 2,000 to 10,000, F above 10,000). Correlation of the area code with log(`Density`) is 0.971. Keep only one in the GLM; the GBM can take both. Log the choice in `DECISIONS.md` |
| `VehBrand` | Categorical, all 11 levels | `B12` kept as its own level |
| `VehGas` | Categorical, 2 levels | Strip the quote marks in the OpenML labels |
| `Region` | Categorical, all 22 levels | |

Every categorical takes its highest-exposure level as the reference. The severity GLM starts from the same list.

### Proposed edits to CLAUDE.md

Proposed only. None has been made.

**Current state**
- E1. Describe what exists: scikit-learn ridge GLMs on seven banded factors, the true Gini history (section 5), single-model lift charts, and the NumPy notebook's location.

**Target end state**
- E2. The GLMs (Poisson frequency, Gamma severity) are fitted with statsmodels, unpenalised. The NumPy IRLS solver is an independent reimplementation that reproduces the statsmodels fit on the pipeline's own design matrix, checked in pytest. It is not the pipeline's engine.
- E3. Add the additive GBM (no interactions) as a diagnostic fourth model.
- E4. No isotonic calibration and no default loadings in the pipeline. One scalar rebalance per model, fitted on training data, with its value reported. A factor far from 1 is investigated, not applied.
- E5. Existing dashboard tabs are rebuilt on the new pipeline. Sliders default to 1.00 and are labelled as a commercial scenario overlay. Dislocation is also shown between models.

**Environment and working rules**
- E6. statsmodels is a pipeline dependency. Resolve the numpy pin. Pin the Python version for Streamlit Cloud.
- E7. Each phase on a local branch, not pushed to `main` until Leo approves, because `main` deploys and the app currently fits models at start-up.

**Phase 1**
- E8. Settle and log the data conventions first: the claim count (options above), a claim-level large-loss cap computed on training claims, and the treatment of exposure and of high claim counts.
- E9. Factor list: all nine freMTPL2 factors for every model, treated as in the table above (GLM banded or continuous, GBM raw). Reference level is the highest-exposure level. `Area` is essentially a banded version of `Density`: check their correlation and keep only one of them in the GLM to avoid collinearity; the GBM can take both. Log the choice in `DECISIONS.md`.
- E10. Replace "existing Gini definition unchanged" with the corrected ordering (sort by predicted rate, ties pooled), with a test that a constant prediction scores zero. The old function stays only in the archived pipeline.
- E11. Frequency metrics are primary: Poisson deviance, frequency Gini, actual versus expected. Pure premium metrics are secondary and are reported with and without the largest claims.
- E12. A new seeded split, stratified on the claim indicator, with the holdout policy IDs saved. All banding, selection and tuning use training data only.
- E13. Re-report the baseline as the statsmodels GLM. The 0.43 and 0.48 figures are retired, not re-reported.
- E14. Move the isotonic and loadings pipeline to `legacy/` rather than deleting it.

**Phase 2**
- E15. Retitle to "From-scratch GLM validated against the pipeline". Remove the solver swap and the "results unchanged" test. Tests compare NumPy with statsmodels on synthetic data and on the pipeline's real training design matrix, for both the Poisson and Gamma models. Phase 2 no longer blocks Phase 3.

**Phase 3**
- E16. The same nine factors for the GLM and the GBM. Add the additive GBM under the same tuning protocol.
- E17. Replace "any calibration step applied identically" with the scalar rebalance.
- E18. Report the holdout deviance difference between models with a paired bootstrap interval.

**Phase 4**
- E19. Interaction terms are fitted in statsmodels. The selection criterion is cross-validated Poisson deviance on the training data. With about 500,000 training rows nearly any interaction is significant under a likelihood-ratio test, so the LR test is reported as supporting evidence only.
- E20. Report two ratios: the share of the total gap closed, and the share of the interaction portion (additive GBM to GBM) closed. Skip a ratio whose denominator is within noise.

**Phase 5**
- E21. The dashboard note reads: GLMs fitted with statsmodels and independently reproduced by a NumPy IRLS solver. The app reads precomputed files only.
- E22. Remove the "Commercial Gini improves the model" framing and correct the Lorenz help text.
- E23. Regenerate the Excel workbook once, for the final GLM only, under a new file name so the hand-built sheets are kept.
- E24. Interview notes: reword question 7 for the new arrangement. Add questions on the claim-count reconciliation, on what the additive GBM showed, and on why claim frequency is not proportional to exposure in this data.

**CV summary**
- E25. Update the models and results sections for the above. Claims 2 (isotonic) and 3 (Gini) describe an earlier version. Claim 4 should say "lift charts" until Phase 3 produces a real double-lift chart.
