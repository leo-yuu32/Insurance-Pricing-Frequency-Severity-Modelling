# Review pack: Poisson GLM against LightGBM on freMTPL2

Prepared on 20 November 2026 for an independent reviewer who has not seen this repository or the conversations in which it was built. It sets out what was done, in what order, and what came out, so that the reviewer can look for weaknesses. It is a technical account, not a summary of strengths.

**Where the numbers come from.** Unless marked otherwise, every figure is read from a file written by the code: `outputs/metrics_final.json`, `outputs/model_comparison.json`, `outputs/bonus_malus_timing.json`, `outputs/gbm_tuning*.json` and `outputs/*_summary.txt`, or from `docs/DECISIONS.md` (entries D1 to D33), which quotes those files and the scripts it names. Figures marked **[computed for this pack]** were computed on 20 November 2026 from the saved predictions in `outputs/predictions.parquet`, without refitting anything. They are not saved in any output file.

**State of the code.** Local branch `phase-5-final-model`, not pushed. The deployed dashboard (https://insurance-pricing-dashboard.streamlit.app/) still runs the legacy version described in section 1. The test suite has 60 tests, all passing on 20 November 2026.

---

## 1. Purpose and context

**What the project is.** A claim frequency and severity model for French motor third-party liability insurance on the public freMTPL2 dataset. Its core is a comparison of a Poisson GLM (statsmodels) with a LightGBM model for claim frequency, on the same data, rating factors, split and metrics. Both use the same Gamma GLM for severity.

**Who built it, and why.** Leo Yu, a final-year mathematics undergraduate at Imperial College London, for applications to graduate pricing analyst and actuarial roles. Interviews are in January 2027. Results from the project may be quoted on his CV and questioned by a pricing actuary in interview.

**How it was produced.**

- An earlier version, written by Leo, used scikit-learn GLMs with a ridge penalty, isotonic calibration and seven hand-set loadings. It reported a Gini improvement from 0.43 to 0.48.
- A read-only audit (`docs/AUDIT.md`, 11 November 2026) found that figure unsupported:
  - the Gini function ranked policies by premium divided by exposure, so a constant prediction scored 0.444;
  - the loadings had been read off charts of the same test set they were scored on;
  - one claim of EUR 4,075,401 was 28.5% of test losses.
- The project was then rebuilt in phases between 11 and 20 November 2026, with the code and documents written by an AI coding assistant (Claude Code) under Leo's direction, as the commit trailers show.
- Decisions recorded as Leo's in `DECISIONS.md` were taken by him. The others were proposed by the assistant and accepted at phase reviews.
- The earlier pipeline is kept, unchanged, in `legacy/`.

**Rules the rebuild set itself** (from the project brief, which is not committed to the repository):

- holdout Poisson deviance is the primary metric;
- all banding, selection and tuning use training data only;
- one scalar rebalancing factor per model, and no other calibration;
- every modelling decision is logged in `docs/DECISIONS.md`;
- results are reported as they come out.

**Phases and commits** (dates are commit dates):

| Phase | Content | Commits |
|---|---|---|
| 0 | Audit of the legacy project | `19981b3`, `8277190` (11 and 12 November) |
| 1 | Data conventions, split, metrics, statsmodels GLM | `193fe80` (12 November) |
| 3 | LightGBM against the GLM | `854364d` (12 November) |
| 4 | SHAP, GLM extensions, additive GBM | `918285d` (12 November) |
| 2 | NumPy IRLS solver | `a55d5ff` (12 November) |
| Review | `BonusMalus` timing checks (D30) | `8a8dfdb` (13 November), `c369f5e` (17 November) |
| 5 | Final specification, then scoring, then dashboard, Excel engine and write-up | `f954934`, `2f9c9da` (18 November), `0cacb8f` (20 November) |

---

## 2. Data

**Source.** freMTPL2freq (OpenML 41214) and freMTPL2sev (OpenML 41215), downloaded with scikit-learn's `fetch_openml`. Raw size: 678,013 policy records, and 26,639 claim records on 24,950 policies.

**Rating factors.** `Area`, `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `VehBrand`, `VehGas`, `Density`, `Region`. Exposure is in years.

**Cleaning, in the order applied** (`src/data.py`):

| Step | Rule | Effect | Data used to set the rule |
|---|---|---|---|
| 1 | Exclude policies with `IDpol` up to 24,951 | 9,387 policies removed, 1.34% of exposure. Every one has a claim count of at least 1 in the frequency file (9,942 claims in total), but the severity file holds only 307 claim records for them. The first policy without a claim is `IDpol` 24,952. | Whole dataset, before the split |
| 2 | Claim count = number of severity records | Changes 23 of the remaining policies (22 with a claim in the frequency file and no record, one with one record fewer). Six `IDpol` values in the severity file (195 records, EUR 0.79 million) have no policy and cannot be used. | Whole dataset |
| 3 | Cap the claim count at 4 | 8 policies (5, 5, 6, 8, 9, 11, 11 and 16 records); 39 claims removed. All 8 are in the training set. | Whole dataset |
| 4 | Cap exposure at 1 | 1,224 policies (the maximum is 2.01). No lower limit: 3,061 one-day policies are kept. | Whole dataset |
| 5 | Cap each claim amount at the 99.9th percentile of training claims | Cap EUR 181,271, from 20,922 training claims; 21 training claims are capped. The amount above the cap is 16.1% of training losses and is reported as a flat large-loss load of 1.1917, not modelled. | Training claims only |

**Rows used.** 668,626 policies, 353,560 policy-years, 26,098 claims (26,137 claim records before step 3).

**Why the block in step 1 is excluded (D1).** The data allow two readings:

- the block was selected because its policies had a claim, so it is not a random sample of the book;
- the claim counts in the block are corrupt, and the severity file holds the true claims.

Exclusion is correct under both readings. Keeping the block and counting its claims from the severity records would be correct under the second only. The two options give portfolio frequencies of 0.0739 and 0.0738. Keeping the file's own claim counts would overstate frequency by 36%.

**Features of the data a reviewer should know about.**

- **Claim frequency is not proportional to exposure.** On the training data:

  | Records | Claims per policy-year |
  |---|---|
  | Full year | 0.054 |
  | Half a year to one year | 0.079 |
  | Under half a year | 0.112 |
  | 0.05 years or less | 0.299 |

  Records under half a year are 50.4% of training policies, 19.6% of training exposure and 29.7% of training claims [computed for this pack].
- **Severity.**
  - The largest claim, EUR 4,075,401, is in the training set (8.3% of training losses, about half of the large-loss load).
  - Fixed amounts recur among training claims: EUR 1,204.00 occurs 3,814 times, 1,128.12 2,354 times and 1,172.00 1,638 times.
- **Duplicate risk profiles.** There are 521,706 distinct combinations of the nine rating factors, and 37.9% of policies share theirs with another policy.
- **`BonusMalus`.** Section 6.

---

## 3. Validation design

**The split.** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on the cleaned data:

- **Groups:** profiles (identical values of all nine rating factors), so twins never straddle the split.
- **Strata:** whether a policy has a claim. 3.687% of policies have a claim in every fold.
- **Folds:** fold 0 is the holdout. Folds 1 to 4 are the training data, and also the four cross-validation folds for every choice made on training data.

| | Policies | Policy-years | Claims | Frequency |
|---|---|---|---|---|
| Training | 534,900 | 282,733 | 20,883 | 0.07386 |
| Holdout | 133,726 | 70,827 | 5,215 | 0.07363 |

**Why the split is grouped.**

- Under an ordinary random split, 32.6% of holdout policies would have an identical-profile twin in training.
- A direct check found no sign that twins share outcomes: among multi-policy profiles, the share with two or more claiming policies is 0.00096, which is what independence predicts. So the grouping is a precaution, not a measured need (D6).

The holdout policy IDs are saved in `artifacts/holdout_idpol.txt`, and a test checks that the code reproduces them. The legacy test set was not reused.

**Seed.** 42 everywhere: the split, the tuning draws, the bootstrap, the SHAP samples and the Excel sample.

**What was chosen on which data.**

| Choice | Data | Method | Entry |
|---|---|---|---|
| GLM bands, factor treatment, `Area` against `Density` | Training | Cut-points from training one-way tables; alternatives compared by 4-fold cross-validated Poisson deviance (`src/compare_glm_designs.py`) | D11 |
| Reference levels | Training | Highest-exposure level of each factor | D11 |
| Severity factors | 19,721 training policies with a claim | 4-fold cross-validated Gamma deviance | D12 |
| Large-loss cap | Training claims | 99.9th percentile | D5 |
| GBM hyperparameters (four variants) | Training | 30-trial random search, the same 30 draws for every variant. 4-fold cross-validation with early stopping after 50 rounds without improvement. The number of trees is the mean stopping round over the folds, and the model is then refitted on all training rows. | D19, D31 |
| GBM encoding of brand and region | Training rows of each fit | Integers ordered by claim frequency | D18 |
| Monotone constraint on `BonusMalus` | Training | Cross-validation of four treatments | D17 |
| SHAP importance and interaction ranking | Training samples of 10,000 and 2,000 rows | `shap.TreeExplainer` | D22, D24 |
| Interaction choice and ladder indicator | Training | 4-fold cross-validated deviance; likelihood-ratio test as supporting evidence only | D23, D24 |
| `BonusMalus` timing checks | Training rows | Descriptive tables; cross-validation with and without the factor | D30 |
| Rebalancing factors | Training | Actual over expected | D13 |
| SHAP plots in `plots/` | Holdout sample of 10,000 | Display only | D22 |
| Model comparison | Holdout | Section 7 metrics | D7 to D9, D21 |

**Search ranges** (D19):

| Parameter | Range |
|---|---|
| Learning rate | 0.03 to 0.2, log scale |
| Leaves | 8 to 64 |
| Minimum data in leaf | 50 to 2,000, log scale |
| L2 penalty | 0.01 to 32, log scale |

**Metrics.**

- **Poisson deviance** (primary): `2 * sum(y log(y / mu) - (y - mu)) / sum(exposure)`, with `mu` = predicted rate times exposure. The holdout null model (training frequency for every policy) scores 0.47703.
- **Gini:**
  - order policies by predicted rate, pooling ties;
  - plot cumulative exposure against cumulative claims;
  - Gini = 1 minus twice the area under that curve, so a constant prediction scores 0 (tested).
- **Actual over expected:** observed claims over predicted claims, overall and by level of a factor.
- **Double lift:** policies sorted by the ratio of GBM to GLM predictions, cut into ten bins of equal exposure.
- **Paired bootstrap:** 1,000 resamples of holdout policies, both models scored on each resample, 2.5th and 97.5th percentiles of the difference (D21). It covers holdout sampling only, not the randomness of the training sample or the tuning.

---

## 4. Full history of holdout use

This section lists, in order, every time holdout data or holdout results were looked at, and every decision they informed.

**Stage A. Before the new holdout existed (Phase 0, 11 to 12 November).**

- **A1. The audit.** It evaluated the legacy pipeline on its own test set, which came from a different random split of differently cleaned data. It also computed outcome statistics on the whole dataset: claim counts by `IDpol`, the reconciliation of the two files, frequency by exposure.
  - Both the legacy test set and the whole-data tables include policies that are now in the holdout. The overlap was not measured.
- **A2. Data rules from whole-data statistics.** Cleaning steps 1 to 4 in section 2 were set from whole-data statistics, before the split. They use the outcome fields (claim counts, claim records) of every policy, the future holdout included. They are data rules applied identically to every model, not model choices.

**Stage B. Nine-factor models scored (Phases 1, 3 and 4; commits of 12 November).**

- **B1. Phase 1.** The nine-factor GLM (63 parameters) was scored: deviance 0.45391, Gini 0.297. No decision followed from the score.
- **B2. Phase 3.** The GBM and the GBM with `BonusMalus` constrained to be increasing were scored (0.44780 and 0.45106), with bootstrap intervals and a double-lift chart. The decision not to constrain was made on training cross-validation before either was scored (D17).
  - One development run scored a GBM that was meant to be unconstrained but still carried the constraint. Its holdout deviance matched the constrained model's to six digits, which exposed the bug. The code was fixed and the model rescored. No modelling choice followed from the values.
- **B3. Phase 4.**
  - **SHAP plots.** Summary and dependence plots were drawn on 10,000 holdout policies. SHAP values depend on the rating factors and the fitted model, not on the holdout claims. The `BonusMalus` dependence plot is cited in D23 as the origin of the ladder indicator. The same sawtooth had already been seen in Phase 3, in a training-data table of frequency by exact `BonusMalus` value (D17).
  - **Models scored:** the GLM with the `BonusMalus` by `Region` interaction (0.45339), the GLM with the ladder indicator (0.44991), the GLM with both (0.44940) and the additive GBM (0.44876).
  - **Ratios on holdout results:** the share of the GLM-to-GBM gap closed and the shape-and-interaction decomposition were computed on holdout deviance and Gini.

**Stage C. Decisions taken with those scores in view (13 to 17 November).**

- **C1. The interaction was dropped (13 November).** A second opinion, accepted by Leo, proposed:
  - making the final GLM the baseline plus the ladder indicator;
  - dropping the interaction because its holdout interval (-0.00001 to 0.00101) included zero;
  - not scoring the alternative interaction, driver age by `BonusMalus`, on the holdout.

  Dropping the interaction is a **selection decision informed by a holdout result**. The alternative interaction was never scored on the holdout.
- **C2. The `BonusMalus` investigation (13 November).** The ladder indicator closing 66% of the gap with one parameter was judged too good to be true, which prompted the `BonusMalus` timing checks (D30). The checks used training rows only. The investigation was **prompted by a holdout result**.
- **C3. `BonusMalus` dropped (17 November).** Leo decided to leave `BonusMalus` out of both final models (D30, D31). The proposal to make the ladder GLM final was withdrawn. The evidence for this decision is training-only, but the route to it starts at C2.

**Stage D. The final models (18 November).**

- **D1. Specification committed (`f954934`, 10:19).** D31 sets out:
  - both specifications, including the GBM parameters tuned on training cross-validation only;
  - the metrics to be reported;
  - the statement to be made if the interval included zero.
- **D2. Scored once (`2f9c9da`, 10:22).** The final GLM and GBM were scored once. A first run of `src/run_final.py` stopped with an error (a missing column) before any holdout metric was computed. It was fixed and run again, and the second run is the one reported.

**Stage E. After the final scoring (18 to 20 November).** Descriptive analyses only. None changed a model.

- The cost of leaving out `BonusMalus`: holdout bootstrap of the nine-factor against the final models.
- Actual over expected by `BonusMalus` band, by rating factor and by record length, for the final models.
- Actual over expected by record length for the nine-factor models. This was computed during the write-up and again for this pack; it is not saved in the outputs.
- 100 holdout policies used to reconcile the Excel engine. Only their rating factors are used; the check is of arithmetic.
- The claim counts behind selected actual-over-expected cells (section 7) [computed for this pack].
- **Choices made after seeing the final results:**
  - the additive GBM was not rebuilt without `BonusMalus` ("no significant gap left to split");
  - log exposure was not added as a covariate (section 9);
  - the wording of the write-up.

**Summary.**

- **Models scored on the holdout:** nine distinct models (seven with nine factors, then the final pair), plus one erroneous run.
- **Holdout-informed decisions:** one selection (dropping the interaction) and one investigation (C2), which led to the change of factor set.
- **What has never touched the holdout:**
  - GLM banding, the severity design and the large-loss cap;
  - all GBM tuning and the monotone decision;
  - the interaction ranking and selection, and the ladder indicator selection;
  - the timing checks and the rebalancing factors.
- **No untouched data remain.**

---

## 5. Models

### Final frequency GLM

Poisson, log link, offset `log(Exposure)`, fitted by statsmodels by maximum likelihood with no penalty. 55 parameters. The bands are those chosen in Phase 1 with `BonusMalus` in the model; they were not chosen again (D31).

| Factor | Treatment | Levels | Reference |
|---|---|---|---|
| `DrivAge` | 10 bands | 18-19, 20-21, 22-23, 24-25, 26-30, 31-40, 41-50, 51-60, 61-75, 76+ | 41-50 |
| `VehAge` | 5 bands | 0-1, 2-5, 6-12, 13-17, 18+ | 6-12 |
| `VehPower` | Categorical | 4, 5, 6, 7, 8, 9, 10, 11, 12+ | 6 |
| `Density` | Continuous | log(`Density`), one coefficient (0.1056) | |
| `VehBrand` | Categorical | 11 brands | B1 |
| `VehGas` | Categorical | Diesel, Regular | Regular |
| `Region` | Categorical | 22 regions | R24 |
| `Area` | Not used | An exact banding of `Density`; its correlation with log(`Density`) is 0.971 (D11) | |
| `BonusMalus` | Not used | Section 6 | |

- Pearson chi-squared over degrees of freedom is 1.6867, so the printed standard errors are too small by a factor of about 1.3 (`outputs/glm_final_frequency_summary.txt`).
- Driver-age relativities against 41-50:

  | Age band | 18-19 | 20-21 | 22-23 | 24-25 | 26-30 | 31-40 |
  |---|---|---|---|---|---|---|
  | Relativity | 3.84 | 2.60 | 1.95 | 1.42 | 1.04 | 0.90 |

- Other relativities: vehicle age 18+ 0.63; diesel 1.18 against regular.

### Final frequency GBM

LightGBM with a Poisson objective. Exposure is never a feature. It enters as `init_score = log(Exposure x training base rate)` (D16):

- `predict` does not add the `init_score` back, so the predicted rate is the base rate times `exp(raw score)`;
- every validation set carries its own `init_score`;
- both points are tested.

**Features:** the eight factors other than `BonusMalus`, raw:

- `Area` as 1 to 6;
- `VehGas` as 0 or 1;
- `VehBrand` and `Region` as integers ordered by training claim frequency (D18).

The integer encoding is used because SHAP interaction values fail on LightGBM's native categorical splits. Its cost against native handling was not measured.

| Learning rate | Leaves | Minimum data in leaf | L2 penalty | Trees | Cross-validated deviance |
|---|---|---|---|---|---|
| 0.1264 | 17 | 1,791 | 13.3636 | 92 | 0.469645 |

- The 30 trials span 0.469645 to 0.470040.
- The best five lie within 0.00003 of each other, with minimum data in leaf from 152 to 1,791.
- 1,791 is the largest value of that parameter among the 30 draws.
- No monotone constraint.

### Severity and pure premium (shared by every frequency model)

**Severity model** (D12):

- Gamma GLM, log link, fitted to each policy's average capped claim amount, with the number of claims as the variance weight.
- One factor: driver age (10 bands), fitted on 19,721 training policies with a claim.
- Dispersion: 13.0.

4-fold cross-validated Gamma deviance:

| Severity design | Cross-validated deviance |
|---|---|
| Intercept only | 1.4171 |
| Driver age only | 1.4116 |
| Any other single factor | 1.4174 to 1.4309 |
| All eight terms | 1.4507 |

**Pure premium.** Frequency times severity, times one rebalancing factor to capped training losses (GLM 1.0017, GBM 1.0022). For metrics on all losses, the flat large-loss load of 1.1917 is applied on top.

### Earlier models: nine factors, set aside

All nine-factor models include `BonusMalus`. In the GLM it is banded: 50, 51-54, 55-64, 65-79, 80-94, 95-99, 100, 101-119, 120+.

| Model | Specification | Cross-validated deviance (training) | Holdout deviance |
|---|---|---|---|
| GLM | 63 parameters | 0.455363 | 0.45391 |
| GLM + `BonusMalus` (3 bands) by `Region` | 105 parameters; selected by cross-validated deviance from SHAP-ranked candidates (D24) | reduction 0.000877 | 0.45339 |
| GLM + ladder indicator | 64 parameters. The indicator is 1 when the score is below 100 and not on the claim-free ladder; relativity 2.21 (D23). | 0.451617 | 0.44991 |
| GLM + both | 106 parameters | | 0.44940 |
| GBM | Learning rate 0.0717, 26 leaves, at least 84 policies per leaf, L2 0.0252, 166 trees | 0.44875 | 0.44780 |
| GBM, `BonusMalus` increasing | Learning rate 0.0402, 33 leaves, at least 779 per leaf, L2 24.34, 242 trees | 0.45175 | 0.45106 |
| Additive GBM | Each tree splits on one factor; learning rate 0.1095, 11 leaves, 418 trees | 0.450151 | 0.44876 |

**Why they were set aside.**

- The training-data evidence in section 6 indicates that `BonusMalus` was recorded after some of the claims it is used to predict.
- The ladder indicator, which marks "has been surcharged for a claim", is the term most exposed to this.
- 95% of the cross-validated GLM-to-GBM gap exists only when `BonusMalus` is in the models (D30).

---

## 6. The `BonusMalus` timing evidence

**Background.**

- **What the dataset documents.** freMTPL2 describes `BonusMalus` only as "between 50 and 350: <100 means bonus, >100 means malus in France". It does not say when the coefficient was recorded relative to the observation period.
- **How the coefficient moves.** It is updated at each annual renewal:
  - a claim-free year multiplies it by 0.95, truncated, down to a floor of 50;
  - an at-fault claim multiplies it by 1.25.
- **The claim-free ladder.** 100, 95, 90, 85, 80, 76, 72, 68, 64, 60, 57, 54, 51, 50.
- **Off-ladder values.** A value below 100 that is off the ladder can only be reached through a claim. For example, 62 is 50 times 1.25. Claim-free renewals then take it to 58, 55, 52 and back to 50.

**The question.** If the coefficient was read at the start of the period, it describes the driver's past and is a legitimate rating factor. If it was read later, a claim in the data may already have moved it, which is target leakage.

All checks use training rows only (`src/check_bonus_malus_timing.py`, `outputs/bonus_malus_timing.json`).

### Check 0: off-ladder share (inconclusive on its own)

- 17.8% of policies with a claim are off the ladder, against 7.9% of policies without one.
- 92.1% of off-ladder policies have no claim in the period.

This is consistent with either reading.

### Check 1: the effect depends on the length of the record

Policies with a score below 100:

| Records | Frequency on the ladder | Frequency off the ladder | Ratio | GLM relativity of the indicator |
|---|---|---|---|---|
| Under half a year | 0.099 | 0.143 | 1.5 | 1.33 |
| Half a year to one year | 0.067 | 0.143 | 2.1 | 1.86 |
| Full year | 0.040 | 0.199 | 5.0 | 3.86 |

- **What it shows.** A driver characteristic should have the same relative effect whatever the length of the record. A coefficient read at the end of a period would produce a larger effect in longer records, because more of the period that triggered the surcharge falls inside the observed record.
- **What it does not show.** The data do not record why a record is short: a mid-term change, a cancellation, new business, or a split at renewal. A record-length dependence could also arise if records are created differently for surcharged policies.

### Check 2: in full-year records, a surcharged value comes close to recording a claim in the year

| Value in full-year records | Policies | Share with a claim |
|---|---|---|
| 50 | 96,733 | 3.6% |
| 51, 54, 57 | 9,177 | 3.0% |
| 62 (surcharged at the last renewal) | 1,326 | 31.7% |
| 58 (one renewal later) | 1,206 | 20.1% |
| 55 (two renewals later) | 1,223 | 11.3% |
| 52 (three renewals later) | 1,135 | 8.6% |
| Other values one surcharge above a claim-free value (63, 67, 71, 75, 106, 112, 118, 125) | 1,997 | 26.8% |

- **What it shows.** A prior-claim effect of the size seen in the newest block of policies (under 2 times; check 3) would put the share at 62 near 7%, not 31.7%. The drivers at 51, 54 and 57 claim less often than those at 50, although they are younger. This is consistent with claimants having been moved off those values.
- **What it does not show.** The 7% benchmark rests on the newest block's estimate, which is based on about 20 claims per value.

### Check 3: the values that carry the excess shift with `IDpol`

Training policies are cut into eight equal blocks by `IDpol`. That `IDpol` follows the order in which policies entered the file is an inference, not documented. Claims per policy-year, all records:

| `IDpol` block | 50 | 62 | 58 | 55 | 52 | Full-year share of records |
|---|---|---|---|---|---|---|
| 1 (lowest `IDpol`) | 0.058 | 0.332 | 0.158 | 0.151 | 0.158 | 33.5% |
| 2 | 0.054 | 0.325 | 0.238 | 0.138 | 0.174 | 40.0% |
| 3 | 0.052 | 0.238 | 0.171 | 0.153 | 0.072 | 29.5% |
| 4 | 0.050 | 0.199 | 0.188 | 0.129 | 0.054 | 20.7% |
| 5 | 0.048 | 0.305 | 0.203 | 0.048 | 0.075 | 22.6% |
| 6 | 0.049 | 0.265 | 0.202 | 0.049 | 0.052 | 14.5% |
| 7 | 0.046 | 0.443 | 0.086 | 0.038 | 0.073 | 36.5% |
| 8 (highest `IDpol`) | 0.052 | 0.090 | 0.101 | 0.044 | 0.053 | 2.3% |

**Cell sizes.** Outside the column for 50, cells hold between 8 and 156 claims. The cells for 55 and 52 in blocks 3 to 8 hold between 8 and 62.

**Part-year records only.**

- In block 8, the frequency at 62 is 0.045 against 0.054 at 50, on 9 claims.
- In blocks 1 to 7, the frequency at 62 is 2.4 to 5.3 times that at 50.

**What it shows.** Read from block 8 back to block 1, the excess appears first at 62 and then also at 58, 55 and 52. Each step back adds one more renewal's worth of surcharge that lines up with a recorded claim. A driver characteristic would not behave like this. A coefficient that already reflects some of the recorded claims would.

**What it does not show.**

- **The staircase is not clean.** In block 7, 62 is at 0.443 and 58 at 0.086. In block 3, 52 is at 0.072.
- **The blocks differ in other ways.** Their full-year shares are not monotone in `IDpol` (block 7 has 36.5%), so `IDpol` order may not be policy age. If the blocks correspond to different source portfolios or extraction years, a block-dependent pattern could have another cause.
- **Only the part-year contrast between block 8 and the rest survives.** The renewal-by-renewal steps for 58, 55 and 52 are too noisy in part-year records alone to quote.

### Cross-validation with and without `BonusMalus` (training)

The GBM keeps its nine-factor parameters here and is not retuned.

| | Null | GLM | GBM | GLM-to-GBM gap |
|---|---|---|---|---|
| Nine factors | 0.47925 | 0.45536 | 0.44875 | 0.00661 |
| Without `BonusMalus` | 0.47925 | 0.47016 | 0.46981 | 0.00035 |

`BonusMalus` is 62% of what the GLM explains in cross-validation, and 69% of what the GBM explains.

### Overall

- **What is not established.** The mechanism. A year-end snapshot of the coefficient would explain check 2, but not the staircase in check 3.
- **No published account.** None was found. One public benchmark repository (TabArena data-foundry, issue 51) notes that its copy of freMTPL2 blanks `BonusMalus` in some `IDpol` blocks, without giving a reason.
- **How the evidence is described.** `DECISIONS.md` calls it strong but unproven. The project treats it as enough to exclude the factor, not as proof of leakage.
- **The original counter-argument was wrong.** D17 first argued that this was not leakage, because 80% of claims below 100 sit on ladder values. That argument is incorrect: it shows only that most claims have not moved the score. D17 now carries a correction.

---

## 7. Results

### Final models (eight factors, holdout)

| | Null | GLM | GBM |
|---|---|---|---|
| Poisson deviance | 0.47703 | 0.46866 | 0.46833 |
| Deviance explained | 0 | 1.75% | 1.82% |
| Gini (claim counts) | 0 | 0.179 | 0.183 |
| Actual / expected | | 0.998 | 0.998 |
| Training deviance | | 0.46954 | 0.46611 |
| Cross-validated deviance (training) | 0.47925 | 0.47016 | 0.46965 |
| Frequency rebalancing factor | | 1.0000 | 0.99996 |

**GLM minus GBM deviance: 0.00033, 95% paired bootstrap interval -0.00019 to 0.00085.**

- The interval includes zero.
- Cross-validation on the training data had predicted a gap of 0.00052. The spread across the 30 tuning trials was 0.000395.
- The holdout is slightly easier than the training folds: its null deviance is 0.47703, against 0.47925 in cross-validation.

### Nine-factor comparison (holdout, set aside)

| Model | Deviance | Deviance explained | Gini | Actual / expected |
|---|---|---|---|---|
| GLM | 0.45391 | 4.85% | 0.297 | 1.000 |
| GLM + interaction | 0.45339 | 4.96% | 0.298 | 1.000 |
| GLM + ladder indicator | 0.44991 | 5.69% | 0.315 | 1.001 |
| GLM + both | 0.44940 | 5.79% | 0.316 | 1.001 |
| Additive GBM | 0.44876 | 5.93% | 0.319 | 1.002 |
| GBM, `BonusMalus` increasing | 0.45106 | 5.44% | 0.310 | 0.999 |
| GBM | 0.44780 | 6.13% | 0.320 | 1.000 |

| Deviance difference | Estimate | 95% interval |
|---|---|---|
| GLM to GBM | 0.00611 | 0.00490 to 0.00732 |
| GLM to constrained GBM | 0.00285 | 0.00185 to 0.00387 |
| GLM to GLM + interaction | 0.00052 | -0.00001 to 0.00101 |
| GLM to GLM + ladder indicator (66% of the gap) | 0.00400 | 0.00313 to 0.00486 |
| GLM + ladder indicator to GBM | 0.00211 | 0.00098 to 0.00327 |
| GLM + both to GBM | 0.00159 | 0.00051 to 0.00270 |
| GLM to additive GBM (84% of the gap) | 0.00515 | 0.00411 to 0.00624 |
| Additive GBM to GBM (16% of the gap) | 0.00096 | 0.00008 to 0.00179 |

**Cost of leaving out `BonusMalus`** (nine-factor against final model, same holdout):

| Model | Deviance cost | 95% interval |
|---|---|---|
| GLM | 0.01475 | 0.01287 to 0.01676 |
| GBM | 0.02053 | 0.01818 to 0.02289 |

### Double lift, final GLM against final GBM (holdout)

Ten bins of about 7,082 policy-years each, sorted by the ratio of the GBM's prediction to the GLM's. Rates are claims per policy-year.

| Bin | GBM / GLM | Actual | GLM | GBM | Closer |
|---|---|---|---|---|---|
| 1 | 0.49 to 0.86 | 0.0757 | 0.0862 | 0.0699 | GBM |
| 2 | 0.86 to 0.91 | 0.0671 | 0.0786 | 0.0698 | GBM |
| 3 | 0.91 to 0.94 | 0.0697 | 0.0762 | 0.0705 | GBM |
| 4 | 0.94 to 0.97 | 0.0745 | 0.0738 | 0.0705 | GLM |
| 5 | 0.97 to 1.00 | 0.0754 | 0.0728 | 0.0716 | GLM |
| 6 | 1.00 to 1.02 | 0.0760 | 0.0717 | 0.0725 | GBM |
| 7 | 1.02 to 1.06 | 0.0690 | 0.0704 | 0.0733 | GLM |
| 8 | 1.06 to 1.10 | 0.0758 | 0.0695 | 0.0748 | GBM |
| 9 | 1.10 to 1.17 | 0.0758 | 0.0696 | 0.0785 | GBM |
| 10 | 1.17 to 1.98 | 0.0772 | 0.0686 | 0.0864 | GLM |

- **No trend in the actual.** The actual frequency stays between 0.067 and 0.077 with no monotone trend, while the GLM falls from 0.086 to 0.069 and the GBM rises from 0.070 to 0.086.
- **Bins and ends.** The GBM is closer in six bins and the GLM in four. In both end bins the actual lies between the two models.
- **Contrast with the nine factors.** In the nine-factor chart (D20), the actual followed the GBM: in the top bin it was 0.124, against 0.075 for the GLM and 0.122 for the GBM.

### Actual over expected by length of record

Holdout policies:

| Records | Share of policies | Share of policy-years | Share of claims | Observed frequency |
|---|---|---|---|---|
| Full year | 25.0% | 47.2% | 36.1% | 0.0563 |
| Half a year to one year | 24.8% | 33.3% | 34.0% | 0.0752 |
| Under half a year | 50.2% | 19.5% | 29.9% | 0.1129 |

Shares and frequencies are [computed for this pack].

Actual over expected:

| Records | Final GLM, holdout | Final GBM, holdout | Final GLM, training | Final GBM, training | Nine-factor GLM, holdout | Nine-factor GBM, holdout |
|---|---|---|---|---|---|---|
| Full year | 0.801 | 0.805 | 0.771 | 0.775 | 0.856 | 0.867 |
| Half a year to one year | 0.994 | 0.990 | 1.044 | 1.040 | 0.962 | 0.955 |
| Under half a year | 1.430 | 1.424 | 1.423 | 1.417 | 1.327 | 1.312 |

The final holdout columns are from `outputs/metrics_final.json`. The training and nine-factor columns are [computed for this pack].

- **Holdout cells.** Full-year records: 1,881 claims against 2,348 expected by the final GLM. Records under half a year: 1,559 against 1,090 [computed for this pack].
- **Training data.** The mis-calibration is equally present on the training data, so it was detectable without the holdout.
- **`BonusMalus` reduces it.** That is consistent with section 6.

### Actual over expected by other factors (holdout)

Full tables are in `outputs/metrics_final.json`. Selected cells, with the claim counts behind them [computed for this pack]:

| Cell | Policy-years | Actual claims | GLM expected | GLM A/E | GBM A/E | (A - E) / sqrt(E), GLM |
|---|---|---|---|---|---|---|
| `BonusMalus` 50 | 44,442 | 2,292 | 3,019.8 | 0.759 | 0.759 | -13.2 |
| `BonusMalus` 100 | 1,177 | 249 | 141.4 | 1.761 | 1.784 | 9.0 |
| `BonusMalus` 120+ | 216 | 100 | 24.4 | 4.099 | 3.985 | 15.3 |
| `VehPower` 9 | 3,026 | 276 | 229.4 | 1.203 | 1.218 | 3.1 |
| `DrivAge` 18-19 | 204 | 47 | 55.7 | 0.844 | 0.893 | -1.2 |
| `Area` F | 1,616 | 149 | 166.1 | 0.897 | 0.978 | -1.3 |
| `VehBrand` B14 | 438 | 12 | 27.7 | 0.433 | 0.442 | -3.0 |

**Reading the last column.**

- It ignores overdispersion. With Pearson chi-squared over degrees of freedom at 1.69, divide it by about 1.3.
- Several dozen cells were inspected, so some values near 3 are expected by chance.
- The GLM balances `VehPower` 9 exactly on the training data (balance property), so the 1.20 is a difference between training and holdout.

The final GLM's actual over expected by `BonusMalus` band rises from 0.76 at 50 to 4.10 at 120 and above: the final model does not price bonus-malus.

### Pure premium (secondary)

| | GLM | GBM |
|---|---|---|
| Gini, capped losses | 0.199 | 0.205 |
| Gini, all losses | 0.207 | 0.212 |
| Gini, all losses without the largest holdout policy | 0.183 | 0.188 |
| Actual / expected, capped losses | 0.974 | 0.973 |
| Actual / expected, all losses after the large-loss load | 0.834 | 0.833 |

The holdout contains few large claims (D14), so the large-loss load, which is driven by training claims, over-predicts there.

---

## 8. Validation of the work itself

**Test suite.** 60 tests, all passing (`python -m pytest`, about two minutes):

| File | Tests | What they check |
|---|---|---|
| `tests/test_data_and_pipeline.py` | 9 | Cleaning rules; claims reconcile with policies; the large-loss cap uses training claims only; the split is grouped and stratified; the holdout matches the saved IDs; the GLMs balance on training data; pure premium rebalancing; predictions positive and complete |
| `tests/test_design_and_glm.py` | 10 | Bands; reference level = highest exposure; the claim-free ladder; interaction columns; the GLM recovers known coefficients on synthetic data; the Poisson GLM reproduces the claim total of every level |
| `tests/test_metrics.py` | 11 | A constant prediction has Gini 0; Gini cannot be raised by ordering by exposure; tie pooling; deviance by hand and against scikit-learn; actual over expected; double-lift bins; bootstrap |
| `tests/test_gbm.py` | 7 | Predicted claims match the observed total on training data; the predicted rate is per policy-year; every dataset carries its own `init_score`; early stopping; the final features exclude `BonusMalus`; the monotone option; the additive model has no interactions |
| `tests/test_irls.py` | 8 | NumPy solver against statsmodels (below) |
| `tests/test_bonus_malus_timing.py` | 5 | Ladder and surcharge values, value groups, record-length classes, frequency tables |
| `tests/test_rating_engine.py` | 5 | Excel band lookups and rating tables reproduce the pipeline; Excel's cached values match Python; the legacy workbook is untouched |
| `tests/test_dashboard_and_summary.py` | 5 | The results file is built from the model outputs; the dashboard imports no modelling library; the app runs headless on the saved results |

**What the tests do not check.**

- The tests check the mechanics that produce the reported figures, not the figures themselves.
- The scripts that write `outputs/` (`run_glm.py`, `run_gbm.py`, `run_interactions.py`, `run_final.py`) are not tested end to end.

**NumPy IRLS solver against statsmodels** (`src/irls.py`, D27 to D29). Test tolerance: relative 1e-6 with absolute 1e-8. Largest differences achieved:

| Design matrix | Coefficients (relative) | Standard errors (relative) | Deviance difference | Iterations, NumPy and statsmodels |
|---|---|---|---|---|
| Final frequency GLM, 534,900 by 55 | 6.0e-12 | 9.1e-9 | 2.9e-11 (of 132,754) | 7 and 7 |
| Nine-factor frequency GLM, 534,900 by 63 | 2.6e-12 | 4.4e-9 | 1.5e-11 (of 128,552) | 7 and 7 |
| Severity GLM, 19,721 by 10 | 1.1e-12 | 6.2e-14 | 3.6e-12 | 7 and 9 |

- The Gamma dispersion is 13.0122399303 in both, using statsmodels' convention (Pearson chi-squared over residual degrees of freedom).
- Two further tests check the algorithm itself:
  - one Poisson IRLS step equals a hand-written Newton-Raphson step to 1e-10;
  - the Gamma log-link IRLS step equals the Fisher scoring step and differs from the Newton-Raphson step.
- The solver is a reimplementation for validation. The pipeline uses statsmodels.

**Excel rating engine** (`artifacts/rating_engine_final_glm.xlsx`, built by `src/build_rating_engine.py`).

- **How it prices.** Excel formulas price a policy from its raw rating factors (`LOOKUP` on band lower bounds, `INDEX`/`MATCH` on the rating table, named ranges).
- **Result.** After recalculation in Excel itself, its Reconcile sheet matches the Python price for 100 of 100 holdout policies. The largest relative difference is 1.6e-15.
- **What it checks.** That the rating table is implemented correctly. It does not validate the model.

---

## 9. Known limitations and open questions

Ranked by how much they affect what the project can claim.

### 1. The final factor set rests on unproven evidence about `BonusMalus`

The final models exclude a factor that every published freMTPL2 comparison uses, on circumstantial evidence (section 6).

- **If the evidence is wrong,** the headline comparison is made on an artificially weakened factor set.
- **If it is right,** every nine-factor figure here is flattered by an unknown amount, and so, possibly, are published benchmarks.
- **Either way,** the final model is not a realistic tariff:
  - deviance explained falls from 4.85% to 1.75%;
  - actual over expected by bonus-malus band runs from 0.76 to 4.10.

  A real insurer knows each driver's coefficient at the start of the policy and would use it.

### 2. The holdout has been reused

- **Scorings.** Nine models plus one erroneous run were scored on the same holdout (section 4).
- **Holdout-informed decisions.** One selection (dropping the interaction) followed a holdout interval. The change of factor set traces back to a holdout result, although the evidence for it is training-only.
- **The final pair.** It was specified and committed before scoring, but the route to its specification passed through holdout results.
- **No untouched data remain.** The size of any optimism cannot be measured.

### 3. Mis-calibration by length of record

Both final models over-predict full-year records (actual over expected 0.80 on the holdout, 0.77 on training for the GLM) and under-predict records under half a year (1.43 and 1.42).

- **Likely cause.** Claim frequency is not proportional to exposure in this data: 0.054 claims per policy-year for full-year records, against 0.112 under half a year and 0.299 at 0.05 years or less. Exposure enters both models as a log offset with its coefficient fixed at 1, which assumes proportionality. Why short records claim more is not known. Possible reasons:
  - policies that end or are re-recorded after a claim;
  - mid-term changes;
  - new business.
- **The usual remedy, and why it was not applied.** The usual remedy is to estimate the exposure effect, for example by fitting log(exposure) as a covariate with its own coefficient instead of a fixed offset. It was not applied, for two reasons:
  - **Before scoring,** D3 had kept exposure out of the models on principle. The observed record length is not known when a policy is quoted, and if short records are partly created by claims, the covariate would learn how records are made.
  - **After scoring,** changing the final models in response to this result would be one more choice made with the holdout in view.
- **Consequences.**
  - The overall level is right (actual over expected 0.998) only because the mix of record lengths is the same in training and holdout.
  - Which frequency an annual policy should be priced at, the full-year experience or the all-records average, is an open question.
  - Both models are affected almost equally, so the GLM-against-GBM conclusion is probably not driven by this, but whether a better exposure treatment would change the gap has not been tested.

### 4. The comparison is not perfectly symmetric

- **Retuning.** After `BonusMalus` was dropped, the GBM was retuned, but the GLM's bands were not re-chosen (they were chosen with `BonusMalus` in the model). This could disadvantage the GLM slightly.
- **GBM settings.**
  - The integer encoding of brand and region was not compared with native categorical handling.
  - The number of trees is the mean early-stopping round, refitted on more data.
  - The chosen minimum data in leaf (1,791) is the largest value drawn, although the best five trials are within 0.00003 of each other.

### 5. Uncertainty is only partly quantified

- **Bootstrap.** The bootstrap covers holdout sampling only. There is one split and one seed, and the final comparison was not repeated across splits.
- **Tuning noise.** The cross-validated gap (0.00052) is of the same order as the spread across tuning trials (0.000395).
- **GLM standard errors.** Overdispersion (Pearson chi-squared over degrees of freedom 1.69) makes the GLM's printed standard errors and likelihood-ratio statistics about 1.3 times too optimistic.

### 6. The severity and pure premium side is weak

- **The model.** Driver age is the only severity factor. The dispersion is 13.0, and recurring fixed claim amounts are not something a Gamma distribution describes.
- **The large-loss load.** It is flat (1.19) and about half of it comes from one EUR 4.08 million training claim. On the holdout, actual over expected on all losses after the load is 0.83.
- **Status.** Pure premium figures are secondary and should not carry conclusions.

### 7. Segment fit on the holdout

`VehPower` 9 is under-predicted by about 20% by both models (276 claims against 229 expected), and `VehBrand` B14 is over-predicted (12 against 28). These may be noise, given the number of cells inspected. They were not investigated further.

### 8. Data provenance

- **The OpenML copy.** The OpenML version of freMTPL2 was used.
- **The excluded block.** The boundary of 24,951 was set from the data (24,500 is the value usually quoted), and what the block is remains unknown.
- **`BonusMalus`.** The source documentation is silent on its timing.

### 9. Minor points

- **Claim count cap.** It applies to frequency, but loss totals use all claim records. This affects 8 policies, all in training.
- **`Area`.** The GBM uses it, but the GLM does not (D11).
- **The dashboard.** It is not deployed, and it shows only the final table, the double-lift chart and a section on the nine-factor comparison.

---

## 10. What I intend to claim on my CV

Copied from `docs/CV_SUMMARY.md`, section 9. The "Do not claim" list is in section 10 of the same file.

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

---

## 11. Questions for the reviewer

The five places where the work seems most open to challenge.

1. **Is the `BonusMalus` evidence enough to exclude the factor?** The evidence is in section 6:
   - the effect depends on record length (relativity 1.33 against 3.86);
   - 31.7% of full-year records at 62 have a claim;
   - the excess shifts with `IDpol` block.

   Is there a reading in which the coefficient was recorded at the start of the period that explains all three? If the timing problem is real, is excluding the whole factor the right response, or is there a defensible partial use?
2. **How much weight can the final comparison bear?** Given the holdout history in section 4, is "on these eight factors the GBM does not beat the GLM" a fair statement? Or should it be weaker, for example "no detectable difference on this holdout"? And is pre-registering the final specification worth anything after the earlier holdout use?
3. **Should the record-length mis-calibration have been fixed before scoring?** Is it a model defect that a log-exposure covariate (or exposure bands) should have corrected before scoring, or a feature of how records are created that a pricing model should not learn? Which frequency should an annual policy be priced at, and could a different exposure treatment change the GLM-against-GBM conclusion?
4. **Is the comparison fair to both models?** Points to weigh:
   - the GLM keeps bands chosen with `BonusMalus` present, while the GBM was retuned;
   - the GBM encodes brand and region as ordered integers;
   - both use one shared severity model and a single split.

   Is either model disadvantaged enough to change the conclusion?
5. **Do any of the CV statements in section 10 go beyond the evidence in sections 2 to 8?** Consider in particular:
   - "identified evidence of target leakage";
   - the statements beginning "Built" and "Wrote", in light of how the work was produced (section 1).
