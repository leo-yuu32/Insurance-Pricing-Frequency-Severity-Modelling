# Modelling decisions

Each entry gives the decision, the alternatives, the reason, and the measured effect where there is one. All choices that needed data were made on the training data only. Numbers come from `outputs/*.json`, `src/compare_glm_designs.py` and the Phase 0 audit.

---

## Phase 1: data conventions

### D1. Exclude the block of policies with `IDpol` up to 24,951

**Decision.** Drop every policy with `IDpol <= 24,951` (9,387 policies).

**What the block is.** Sorted by `IDpol`, every one of the first 9,387 policies has `ClaimNb >= 1` in the frequency file. The first policy with no claim is `IDpol` 24,952, and from there on the file looks normal (about 3.7% of policies have a claim). Inside the block only 307 claim records exist in the severity file, against 9,942 claims in the frequency file. The boundary usually quoted for this dataset is 24,500; the data put the end of the block at 24,951, and the 248 policies between the two values show the same pattern, so the cut is made where the pattern stops.

**Alternatives.**

| | A: exclude the block (chosen) | B: keep it, count claims from the severity records | C: keep the file's `ClaimNb` |
|---|---|---|---|
| Policies | 668,626 | 678,013 | 678,013 |
| Exposure (policy-years) | 353,560 | 358,360 | 358,360 |
| Claims | 26,137 | 26,444 | 36,102 |
| Frequency | 0.0739 | 0.0738 | 0.1007 |
| Frequency inside the block | not used | 0.0640 | 2.0712 |

**Cost of A.** 1.34% of exposure and 307 claims that do have amounts (1.2% of claim records).

**Effect on frequency and on actual versus expected.** A and B give almost the same portfolio frequency (0.0739 against 0.0738, a 0.2% difference), so the choice between them moves the overall level of any model by about 0.2%. Under B the block would sit at 0.064, which is 13% below the rest of the book, so a model fitted to everything would show actual over expected of roughly 0.87 on those policies before allowing for their rating factors. Under C the frequency is overstated by 36% and frequency times severity overstates losses by the same amount; this is what the legacy pipeline did.

**Why A.** There are two readings of the block, and the data do not settle which is true.

- Reading 1: these are policies selected because they had a claim, with the amounts missing. Then they are not a random sample of the book, and treating them as claim-free (B) is wrong.
- Reading 2: these are ordinary policies whose `ClaimNb` field is corrupted, and the severity file holds their true claims. The severity-based claim rate inside the block (3.1% of policies, frequency 0.064) is close to the rest of the book (3.7%, 0.074), which supports this reading. Then B is right and A loses 1.3% of good data.

A is correct under both readings. B is correct only under the second. The cost of A is small and its effect on the level is 0.2%, so A is the safer choice. B is the convention I believe the published treatments of this dataset use; a third-party project reports that it reproduces the policy and claim counts in Wüthrich and Merz (2023), which I have not checked against the book.

**Did I disagree?** No. My Phase 0 recommendation was B. After measuring, the two are not materially different, and A is the more defensible.

### D2. For the remaining policies, the claim count is the number of severity records

**Decision.** `ClaimNb` is the number of rows the policy has in `freMTPL2sev`.

**Why.** Frequency and severity then describe the same claims, so frequency times severity reproduces total losses with no correction factor. Outside the excluded block this changes 23 policies out of 668,626 (22 with a claim in the frequency file and no record, one with one record fewer), which is 23 claims out of 26,160.

**Also.** Six `IDpol` values in the severity file (195 rows, EUR 0.79 million) do not exist in the frequency file. They have no rating factors and cannot be used.

### D3. Exposure: clip at 1, keep the one-day policies

**Decision.** Exposure above 1 is set to 1 (1,224 policies; the maximum in the file is 2.01). No lower limit is applied.

**Alternative.** The legacy pipeline dropped exposures of 0.003 or less, calling them "likely data errors". Those are exactly the one-day policies (1/365 = 0.0027). There are 3,061 of them after D1 and nothing shows they are errors.

**Why.** With exposure as an offset, a short policy is handled correctly by the model: its expected claim count is simply small. Dropping rows on the size of the offset needs a reason, and there is none.

**Something to know for interviews.** In this dataset claim frequency is not proportional to exposure. On the legacy test set, policies with exposure up to 0.05 had an observed frequency of 1.04 claims per policy-year, against 0.07 for full-year policies. The Poisson model with a log-exposure offset assumes proportionality, so every model here inherits that misfit equally. Exposure is never used as a feature, because a model that learned "short exposure means high risk" would be learning an artefact of how policies are recorded, not something a rating factor should carry.

### D4. Claim counts above 4 are set to 4

**Decision.** `ClaimNb = min(records, 4)`.

**Effect.** Eight policies are affected (5, 5, 6, 8, 9, 11, 11 and 16 records), removing 39 claims out of 26,137 (0.15%).

**Why.** A private motor policy with 16 claims in four months is almost certainly a fleet or a data error. One such policy would also dominate a deviance comparison between models if it fell in the holdout. Capping at 4 follows the scikit-learn worked example on this dataset. The legacy pipeline dropped policies with 10 or more; capping keeps the policies and their exposure.

### D5. Large losses: cap individual claims at the 99.9th percentile of training claims

**Decision.** Each claim amount is capped at EUR 181,271, the 99.9th percentile of the 20,922 claims on training policies. The cap is computed from training claims only and then applied to all claims. 21 training claims are capped.

**Alternative.** The legacy pipeline capped the policy-level average severity (total loss divided by claim count). That mixes two things: a policy with one large and one small claim is treated differently from a policy with the large claim alone.

**What is done with the excess.** The severity model is fitted to capped amounts, so the modelled pure premium is on a capped basis. The amount above the cap is 16.1% of training losses. It is reported as one flat **large-loss load of 1.192** (training losses divided by capped training losses) and is not spread by rating factor, because 21 claims cannot support a model of who has large losses.

**Why frequency metrics are primary.** The largest claim in the dataset is EUR 4,075,401, on a 19-year-old. It is one claim out of 26,137.

- In the Phase 0 audit it sat in the legacy test set, where it was 28.5% of all test losses. Any change that moved young drivers up the ranking captured it, and that single policy moved the legacy Gini by about 0.18.
- In the new split it falls in the training set, where it is 8.3% of training losses and accounts for about half of the large-loss load.

A metric computed on loss amounts depends on where that one claim lands. A metric computed on claim counts does not. That is why every comparison in this project is made first on claim frequency (Poisson deviance, frequency Gini, actual versus expected), and why pure premium figures are secondary and are quoted on capped losses.

### D6. The split: five folds, grouped by risk profile, stratified on the claim indicator

**Decision.** Policies are cut into five equal folds with `StratifiedGroupKFold(shuffle=True, random_state=42)`. Fold 0 is the holdout (133,726 policies). Folds 1 to 4 are the training data (534,900 policies) and are also the cross-validation folds for every choice made on training data: the GLM design, the GBM tuning and the interaction selection.

| | Policies | Exposure | Claims | Frequency |
|---|---|---|---|---|
| Train | 534,900 | 282,733 | 20,883 | 0.07386 |
| Holdout | 133,726 | 70,827 | 5,215 | 0.07363 |

**Grouped.** A "profile" is a set of policies with identical values of all nine rating factors. There are 521,706 profiles; 37.9% of policies share theirs with at least one other policy. Under an ordinary random split, 32.6% of holdout policies would have a twin in the training set. All policies of a profile are kept on the same side.

**Is the leak material?** A simple check found no sign that it is: among multi-policy profiles, the share with two or more claiming policies is 0.00096, which is what independence predicts. CLAUDE.md says to group only if the effect is material, so on that evidence alone grouping is not required. I grouped anyway, for two reasons. Published comparisons on this dataset report that duplicate policies inflate the gains of flexible models, which is exactly the comparison this project makes. And grouping costs nothing, whereas "a third of your holdout has an identical twin in training" is a question that is better not to have to answer.

**Stratified.** The share of policies with a claim is 3.687% in every fold.

**New, not reused.** The legacy test set was used to choose the loadings, so it is not a clean holdout. The holdout policy IDs are saved in `artifacts/holdout_idpol.txt`, and a test checks that the code still produces the same holdout.

---

## Phase 1: metrics

### D7. Poisson deviance on the holdout is the primary metric

**Decision.** Models are compared on the exposure-weighted mean Poisson deviance: the total deviance of the claim counts divided by total exposure.

`D = 2 * sum( y log(y / mu) - (y - mu) ) / sum(exposure)`, with `mu = predicted rate * exposure`.

**Why.** It is the loss the Poisson GLM and the Poisson GBM both minimise, so it compares them on their own terms. It is a proper scoring rule: it rewards getting the level right as well as the ordering. Gini only measures ordering.

**How to read it.** The absolute value means little. The null model (every policy at the training frequency) scores 0.47703 on the holdout. Differences between models are small numbers, which is normal for claim counts: most of the deviance is irreducible Poisson noise.

### D8. Gini: sort by predicted rate, pool ties

**Decision.** Order policies from lowest to highest predicted rate. Plot cumulative exposure against cumulative actual (claims or losses). Gini is one minus twice the area under that curve. Policies with the same prediction are pooled into one point.

**What was wrong before.** The legacy function sorted by prediction divided by exposure, although the prediction was already a rate. Policies were ranked mostly by 1 / exposure, and a constant prediction scored 0.444. Tests now check that a constant prediction scores exactly zero and that the result cannot be changed by reordering by exposure.

### D9. Actual versus expected, and double lift

- **Actual versus expected** is observed claims over predicted claims, overall and by level of a rating factor. A model can rank well and still be at the wrong level for a segment.
- **Double lift** compares two models directly. Policies are sorted by the ratio of model B's prediction to model A's and cut into ten bins of equal exposure. In each bin the chart shows the actual rate and both predictions. Where the models disagree most (the end bins), the one closer to the actual line is the better one.

---

## Phase 1: the GLMs

### D10. statsmodels, unpenalised

**Decision (Leo's).** Both GLMs are fitted with statsmodels by ordinary maximum likelihood. The NumPy IRLS solver is a separate reimplementation, validated against statsmodels, and is not the pipeline's engine.

**Why unpenalised.** The legacy pipeline used scikit-learn with a ridge penalty (`alpha = 0.01`). The penalty shrank every driver-age coefficient towards the reference level, so ages 18 to 21 were priced at the portfolio average frequency (0.101 predicted, 0.229 observed on the training data). An unpenalised GLM cannot do that, because of the balance property.

**The balance property.** For a Poisson GLM with a log link, the score equations are `X'(y - mu) = 0`. The column of `X` for a dummy variable is 1 for the policies at that level and 0 elsewhere, so that equation says: predicted claims equal observed claims for that level. With an intercept the same holds for the total, and therefore for the reference level too. So on its training data the model reproduces the claim count of every level of every categorical factor exactly. `tests/test_data_and_pipeline.py::test_frequency_glm_balances_on_training_data` checks this on the fitted model, and it is why the frequency rebalancing factor of the GLM is 1.000 by construction.

**Overdispersion.** The Pearson chi-squared statistic divided by its degrees of freedom is 1.72 for the frequency model. Under a true Poisson model it would be near 1. The coefficient estimates are unaffected, but the standard errors printed by statsmodels are too small by a factor of about 1.3 (the square root of 1.72). Read the p-values with that in mind.

### D11. Rating factors and how the GLM uses them

All nine freMTPL2 factors are available to every model. The GBM takes them raw. The frequency GLM uses eight terms and 63 parameters:

| Factor | GLM treatment | Levels |
|---|---|---|
| `DrivAge` | Banded | 18-19, 20-21, 22-23, 24-25, 26-30, 31-40, 41-50, 51-60, 61-75, 76+ |
| `VehAge` | Banded | 0-1, 2-5, 6-12, 13-17, 18+ |
| `VehPower` | Categorical | 4, 5, 6, 7, 8, 9, 10, 11, 12+ |
| `BonusMalus` | Banded | 50, 51-54, 55-64, 65-79, 80-94, 95-99, 100, 101-119, 120+ |
| `Density` | Continuous | log(`Density`), one coefficient |
| `Area` | Not used | see below |
| `VehBrand` | Categorical | all 11 brands, `B12` separate |
| `VehGas` | Categorical | Diesel, Regular |
| `Region` | Categorical | all 22 regions |

Cut-points were set from one-way frequency tables on the training data. The alternatives were then compared by 4-fold cross-validated Poisson deviance on the training data (`python src/compare_glm_designs.py`; lower is better):

| Design | CV deviance | Against the chosen design |
|---|---|---|
| Chosen design | 0.455363 | |
| `BonusMalus` linear, capped at 150 | 0.457748 | +0.002385 |
| `Area` in place of log(`Density`) | 0.455374 | +0.000011 |
| `Area` as well as log(`Density`) | 0.455356 | -0.000007 |
| Legacy eight driver-age bands | 0.455446 | +0.000083 |
| Legacy factor set (no `BonusMalus`, no `Density`) | 0.470396 | +0.015033 |
| Intercept only | 0.479249 | +0.023886 |

What this shows:

- **`BonusMalus` is banded, not linear.** This changes what the Phase 0 audit proposed. The training one-way is not log-linear: frequency is 0.051 at 50, jumps to about 0.10 for 55 to 64, is flat to about 94, then climbs steeply to 0.21 at exactly 100 and 0.45 from 120. A straight line cannot follow that, and it costs 0.0024 of deviance, which is 10% of everything the model explains.
- **`Area` is dropped.** `Area` is an exact banding of `Density` (A is 1 to 50, B 50 to 100, C 100 to 500, D 500 to 2,000, E 2,000 to 10,000, F above 10,000); the correlation between the area code and log(`Density`) is 0.971. Using `Area` in place of log(`Density`) is worse by 0.00001 and adding it on top improves by 0.000007: both differences are noise. One coefficient on log(`Density`) is kept and five dummies are saved. The GBM is given both, which is harmless for trees.
- **Ten driver-age bands, not eight.** This also changes the audit's proposal. Within the old 18 to 21 band the training frequency is 0.278 at 18 to 19 and 0.193 at 20 to 21; within 22 to 25 it is 0.147 and 0.110. Splitting both bands improves the CV deviance by 0.00008.
- **`BonusMalus` and `Density` matter most.** The legacy seven factors explain 0.0089 of deviance in cross-validation; the nine factors explain 0.0239. Giving these two factors to the GBM and not to the GLM would have made the comparison meaningless.
- **Vehicle age 0 is not special.** The audit expected a separate band for new vehicles. On the cleaned data the frequency at age 0 (0.072) is the same as at ages 1 to 3 (0.070 to 0.075). The large "new car" effect in the legacy model came from the excluded block.

**Reference levels.** Each categorical factor's reference is its highest-exposure level in the training data (for example 41 to 50 for driver age, `BonusMalus` 50, region R24). Coefficients are then relative to the most common policy, and the intercept is estimated from the most data. In an unpenalised GLM the choice of reference does not change the fitted values, only how the coefficients read.

**A point worth understanding.** The fitted driver-age relativities are not monotone: 0.57 for 26 to 30 and 0.70 for 31 to 40, against 1.00 for 41 to 50. On a one-way table those ages are not lower risk. The explanation is `BonusMalus`. A young driver cannot have a low bonus-malus score yet, so young drivers carry high scores and the model has already charged them through that factor; the age coefficient is what is left over. This is the same structure that makes driver age with bonus-malus a likely interaction in Phase 4.

### D12. Severity: a Gamma GLM with a log link, and only driver age

**Decision.** The response is the policy's average capped claim amount, with the number of claims as the variance weight (the mean of n Gamma claims has variance divided by n). The only rating factor is banded driver age (10 parameters).

**Evidence.** 4-fold cross-validated Gamma deviance per claim on the 19,721 training policies with a claim (lower is better):

| Severity design | CV deviance | Against intercept only |
|---|---|---|
| Intercept only | 1.4171 | |
| Driver age only | 1.4116 | -0.0055 |
| Any other single factor | 1.4174 to 1.4309 | worse |
| All eight terms | 1.4507 | +0.0336 |

Driver age is the only factor that predicts severity out of sample. Every other factor makes the predictions worse, and using all of them is much worse than using none. The fitted effect is that claims of drivers aged 18 to 19 cost about 2.4 times those of drivers aged 41 to 50.

**Why severity is so hard to predict here.** The dispersion estimate is 13.0 (Pearson chi-squared over degrees of freedom), so the coefficient of variation of a claim is about 3.6 even after capping. Many claims are also fixed amounts: EUR 1,204.00 occurs 3,814 times among training claims, EUR 1,128.12 occurs 2,354 times and EUR 1,172.00 occurs 1,638 times. These look like settlements under a fixed-amount convention between insurers. A Gamma distribution cannot describe a spike at fixed values, so the model is only a description of the mean.

**Shared by every model.** The same severity model is used with every frequency model, so differences in pure premium between models come from frequency alone.

### D13. One rebalancing factor per model, reported

**Decision.** After fitting, each model's predictions are multiplied by one number: training actual over training expected.

| | Factor | Comment |
|---|---|---|
| Frequency GLM | 1.0000 | exactly 1, by the balance property |
| Pure premium (frequency times severity), capped basis | 1.0017 | the Gamma log-link model balances relative errors, not amounts, so a small factor is expected |
| Large-loss load | 1.1917 | not a rebalance: the share of losses above the claim cap (D5) |

**Why this replaces isotonic calibration.** The legacy pipeline needed an off-balance factor of 0.816 and then an isotonic step. The 0.816 was the claims without amounts (D1) and the capped severity partly cancelling. Once the data are consistent, the factor is 1.002 and there is nothing left to calibrate. A factor far from 1 is a symptom to investigate, not a number to apply.

### D14. Baseline results on the holdout

Statsmodels GLM, 63 frequency parameters, fitted on 534,900 training policies, scored once on the 133,726 holdout policies.

| Frequency | Value |
|---|---|
| Poisson deviance | 0.45391 |
| Null model deviance | 0.47703 |
| Deviance explained | 4.85% |
| Gini (claim counts) | 0.297 |
| Actual over expected | 0.9997 |

| Pure premium (secondary) | Value |
|---|---|
| Actual over expected, capped losses | 0.975 |
| Gini, capped losses | 0.335 |
| Gini, all losses | 0.341 |
| Actual over expected, all losses, after the large-loss load | 0.835 |

The last figure shows the large-loss problem from the other side: the load of 1.19 is driven by training-set claims including the EUR 4.08 million one, and the holdout happens to contain almost no large claims (only 2% of its losses are above the cap).

**These do not continue the legacy figures.** The legacy 0.43 and 0.48 were pure premium Ginis from a function that ranked by exposure, on data that included the anomalous block. They are retired. The figures above are on different data, a different split and a corrected definition.

### D15. Practical choices

- **`requirements-dev.txt`** pins the modelling environment (statsmodels, lightgbm, shap, pytest and the versions of numpy, pandas and scikit-learn that produced these numbers). `requirements.txt` is left as it is, because it is what the deployed app installs and the app does not use the new code yet.
- **The legacy pipeline stays in place** (`pricing_engine.py`, `frequency_model.py`, `severity_model.py`, `data_preprocessing.py`, `dashboard.py`) so the deployed app keeps working. It moves to `legacy/` when the dashboard is rebuilt.
- **Caches.** The raw data and the fold assignment are cached in the git-ignored `data/` folder. Computing the grouped, stratified folds takes about four minutes; with the cache, `python src/run_glm.py` takes under a minute.

---

## Phase 3: the LightGBM frequency model

### D16. Poisson objective, with exposure as an offset through `init_score`

**Decision.** LightGBM with `objective = "poisson"`. The target is the claim count. Exposure is not a feature. It enters as an offset:

`init_score = log(Exposure) + log(base_rate)`, where `base_rate` is the training claim frequency.

**Why an offset.** It is the same model structure as the GLM: expected claims = exposure times a rate that depends on the rating factors. The trees then model the log of the rate relative to the base rate.

**Why `base_rate` is included.** When an `init_score` is supplied, LightGBM starts boosting from it and does not add its own starting constant. With only `log(Exposure)`, the first trees would be spent moving every policy from a rate of 1 claim per year down to about 0.07. Putting the base rate in the offset means the trees start from the right level.

**The two traps, and how the code avoids them.**

1. `predict` returns the trees' contribution only and does not add the `init_score` back. The predicted rate is `base_rate * exp(raw_score)`, and the predicted claim count is that times exposure. `gbm.predict_rate` is the only place this is done. A test checks that predicted claims on the training data sum to the observed total.
2. A validation set used for early stopping needs its own `init_score`, or it is scored as if every policy had one year of exposure. Every `Dataset` is built by the same function (`gbm.make_dataset`), so none can be created without it. A test checks the validation set's `init_score`.

### D17. No monotone constraint on `BonusMalus`

**Reviewed and left closed (Leo, 17 November 2026).** The sawtooth that makes the constraint expensive may be partly a data artefact (D30). The question does not arise for the final model, which does not use `BonusMalus` (D31).

**Decision.** The primary GBM has no monotone constraint. A constrained version is tuned and reported alongside it, so the cost of the constraint is on record.

**What was expected.** A higher bonus-malus score should mean a higher claim frequency, so an increasing constraint looked like a free way to make the GBM easier to defend.

**What cross-validation on the training data showed.** Same parameters, four ways of treating `BonusMalus` (`python src/compare_gbm_constraints.py`):

| Constraint on `BonusMalus` | CV deviance | Against no constraint |
|---|---|---|
| None | 0.448758 | |
| Increasing, method `basic` (LightGBM default) | 0.451751 | +0.002993 |
| Increasing, method `intermediate` | 0.451743 | +0.002985 |
| Increasing, method `advanced` | 0.451664 | +0.002906 |

The constraint costs 0.0030 of deviance, whichever method enforces it. For scale, the whole gain of the constrained GBM over the GLM in cross-validation is 0.0036. So this is not a quirk of one algorithm: the effect of `BonusMalus` is not monotone.

**Why it is not monotone.** The French bonus-malus coefficient starts at 100. Each claim-free year multiplies it by 0.95 (truncated), down to a floor of 50, which gives a fixed "claim-free ladder": 100, 95, 90, 85, 80, 76, 72, 68, 64, 60, 57, 54, 51, 50. Each at-fault claim multiplies it by 1.25. A value below 100 that is not on the ladder can therefore only be reached by having had a claim. For example 62 is 50 times 1.25: a driver at the maximum bonus who then had a claim.

Training frequencies by exact value show it:

| `BonusMalus` | 57 | 58 | 60 | 62 | 63 | 64 | 68 | 72 | 75 | 76 | 77 | 80 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| On the ladder | yes | no | yes | no | no | yes | yes | yes | no | yes | no | yes |
| Frequency | 0.056 | 0.172 | 0.068 | 0.283 | 0.158 | 0.066 | 0.071 | 0.077 | 0.240 | 0.083 | 0.372 | 0.095 |

Across all values below 100, the off-ladder values hold 8% of training exposure and 20% of the claims (frequency 0.167, against 0.066 for all other policies). The curve is a sawtooth, and a model forced to be increasing has to flatten it. A driver at 62 is riskier than a driver at 64, and the score says so.

**Is this leakage?** It would be if the recorded score had already been updated for the claims being predicted. It has not: 80% of the claims below a score of 100 happen to policies on ladder values, which a policy could not hold after an at-fault claim in the same period. The score describes the past.

**Correction (13 November 2026).** That argument was wrong. That most claims sit on ladder values shows only that most claims have not moved the score. It says nothing about the claims that have. D30 tests the question directly and finds strong, unproven evidence that some have.

**Why not keep the constraint anyway.** A pricing team might: "a worse score never lowers the price" is easy to explain to a customer. But the purpose of the GBM in this project is to measure what the data support, so the benchmark is the unconstrained model. The constrained model is reported as what a team would give up for that rule.

**Effect on the holdout.** Unconstrained 0.44780, constrained 0.45106 (D20). This was read after the decision, which was made on the cross-validation table above.

### D18. The GBM's features

All nine rating factors, raw: `VehPower`, `VehAge`, `DrivAge`, `BonusMalus` and `Density` as numbers; `Area` as 1 to 6 (it is ordered, A is rural and F is urban centre); `VehGas` as a 0/1 flag.

**Brand and region are encoded as integers ordered by training claim frequency**, not as LightGBM categorical features.

- *Why.* SHAP interaction values, which Phase 4 needs, cannot be computed for a LightGBM model with native categorical splits (the SHAP library fails on them). With every split numeric, they can.
- *Is the encoding legitimate?* The order is computed from the rows the model is being fitted on, inside each cross-validation fold as well, so no validation or holdout information enters it. A tree can still separate any group of brands that are adjacent in the order.
- *Cost.* Not measured against native categorical handling.

### D19. Tuning

**Decision.** Random search, 30 trials, scored by 4-fold cross-validated Poisson deviance on the training data, using the same grouped, stratified folds as every other choice. Each fit stops when the validation fold's loss has not improved for 50 rounds.

| Parameter | Range searched | Chosen |
|---|---|---|
| `learning_rate` | 0.03 to 0.2 (log scale) | 0.0717 |
| `num_leaves` | 8 to 64 | 26 |
| `min_data_in_leaf` | 50 to 2,000 (log scale) | 84 |
| `lambda_l2` | 0.01 to 32 (log scale) | 0.0252 |
| Number of trees | mean early-stopping round over the four folds | 166 |

The final model is fitted once on all training rows with these parameters and 166 trees. The parameters live in `outputs/gbm_tuning.json`, which `run_gbm.py` reads; every trial is logged there.

**How much the tuning mattered.** The 30 trials span 0.44875 to 0.44930 in CV deviance, a range of 0.0006. The gap to the GLM (0.4554) is ten times that. The result does not depend on finding the right settings.

**What the parameters do** (for the interview notes): the learning rate scales each tree's contribution, so a lower rate needs more trees and usually generalises a little better; `num_leaves` sets how complex one tree can be, and so how high an order of interaction it can represent; `min_data_in_leaf` stops a leaf being fitted to a handful of policies, which matters when only 4% of policies have a claim; `lambda_l2` shrinks leaf values towards zero.

### D20. GLM versus GBM on the holdout

**Superseded as a pricing result (D30 to D32).** The figures below are correct as computed and stay in the results as a labelled comparison. But the gap runs almost entirely through `BonusMalus`, which appears to be recorded after some of the claims it predicts, so the reading below (including "the commercial meaning of the gap") should not be quoted as a finding about GLMs and GBMs.

Scored once, on the 133,726 holdout policies.

| Frequency model | Poisson deviance | Deviance explained | Gini | Actual / expected |
|---|---|---|---|---|
| Null (training frequency) | 0.47703 | 0 | 0 | |
| GLM (statsmodels, 63 parameters) | 0.45391 | 4.85% | 0.297 | 0.9997 |
| GBM, `BonusMalus` constrained | 0.45106 | 5.44% | 0.310 | 0.9994 |
| **GBM** | **0.44780** | **6.13%** | **0.320** | **1.0000** |

| Difference in deviance from the GLM | Estimate | 95% bootstrap interval |
|---|---|---|
| GBM | 0.00611 | 0.00490 to 0.00732 |
| GBM, `BonusMalus` constrained | 0.00285 | 0.00185 to 0.00387 |

**Reading.** The GBM is better than the GLM, and the interval is well clear of zero. In relative terms the GBM explains 6.1% of the null deviance against the GLM's 4.9%: about a quarter more. The GLM already captures four fifths of what the GBM finds.

**Overfitting check.** Deviance on training data against holdout: GLM 0.45468 and 0.45391; GBM 0.44147 and 0.44780. The GBM fits its training data more closely than it predicts new data, as expected, but its holdout score agrees with its cross-validation score (0.44875), which is the estimate that was used to choose it.

**Rebalancing factors.** GBM frequency 0.9998; pure premium 1.0026. Both models are at the right overall level without adjustment.

**The double-lift chart** (`plots/double_lift_glm_vs_gbm.png`). Policies sorted by GBM prediction over GLM prediction, ten bins of equal exposure:

| Bin | GBM / GLM | Actual | GLM | GBM |
|---|---|---|---|---|
| 1 | 0.26 to 0.74 | 0.0645 | 0.1028 | 0.0670 |
| 2 | 0.74 to 0.82 | 0.0662 | 0.0774 | 0.0607 |
| 5 | 0.92 to 0.97 | 0.0654 | 0.0696 | 0.0657 |
| 6 | 0.97 to 1.02 | 0.0681 | 0.0694 | 0.0690 |
| 9 | 1.18 to 1.31 | 0.0775 | 0.0661 | 0.0817 |
| 10 | 1.31 to 14.2 | 0.1240 | 0.0747 | 0.1215 |

Where the models agree (bins 5 and 6) both are right. Where they disagree, the actual frequency follows the GBM: in bin 1 the GLM predicts 0.103 and the actual is 0.065; in bin 10 the GLM predicts 0.075 and the actual is 0.124. A GLM-priced book would be undercharging the tenth of the portfolio in bin 10 by about 40%, and an insurer pricing with the GBM would not want those policies at the GLM's price. That is the commercial meaning of the gap.

**Pure premium (secondary).** With the shared severity model, Gini on capped losses is 0.335 for the GLM and 0.348 for the GBM; actual over expected on capped losses is 0.975 for both.

**Small cells.** Brand `B14` shows actual over expected of 0.42 on the holdout under both models. It has about 450 policy-years there, so that is a dozen claims against an expected thirty: noise, not a model fault.

### D21. The bootstrap interval

**Decision.** The difference in holdout deviance between two models is reported with a paired bootstrap: holdout policies are resampled with replacement 1,000 times, both models are scored on each resample, and the 2.5th and 97.5th percentiles of the difference are reported.

**Why paired.** Both models are scored on the same policies, and their errors are strongly correlated (a policy with an unlucky claim hurts both). Resampling the pair keeps that correlation, so the interval is for the difference, which is much tighter than comparing two separate intervals.

**What it does and does not cover.** It answers "would this difference survive a different holdout sample?". It does not cover the randomness of the training sample or of the tuning.

---

## Phase 4: explaining the GBM and feeding it back into the GLM

### D22. SHAP on the GBM

**What a SHAP value is.** For one policy, the GBM's prediction on the log scale is a base value plus one number per rating factor. That number is the factor's SHAP value: how much this policy's value of the factor moves its log frequency, averaged over the orders in which the factors could be added. A SHAP value of 0.69 doubles the predicted frequency. A SHAP interaction value splits each pair's joint contribution out of the two main effects, so the prediction becomes main effects plus pairwise terms.

**Setup.** `shap.TreeExplainer` on the tuned GBM. Importance uses 10,000 training rows; interaction values use 2,000 training rows (they cost far more to compute). Both samples are from the **training** data, because the ranking is used to choose GLM terms. The plots in `plots/` use a holdout sample and are for display only. The SHAP values add up to the model's raw prediction to within 1e-14, which checks that the explainer reads the model correctly.

**Importance** (mean absolute SHAP value, log scale):

| `BonusMalus` | `Density` | `DrivAge` | `Region` | `VehGas` | `VehAge` | `VehBrand` | `VehPower` | `Area` |
|---|---|---|---|---|---|---|---|---|
| 0.352 | 0.116 | 0.111 | 0.080 | 0.079 | 0.079 | 0.060 | 0.030 | 0.001 |

`BonusMalus` matters three times as much as the next factor. `Area` contributes nothing once `Density` is present, which confirms D11 from the GBM's side.

### D23. The bonus-malus ladder indicator

**Under review (13 November 2026).** D30 gives evidence that the pattern this term picks up is partly the claims being predicted, not prior history. Do not quote this entry without D30.

**Decision.** Offer the GLM one new main-effect term: `BonusMalusOffLadder`, equal to 1 when the score is below 100 and is not one of the claim-free ladder values (D17), and 0 otherwise.

**Where it came from.** The SHAP dependence plot for `BonusMalus` (`plots/shap_dependence_BonusMalus.png`) is not a smooth curve. It has spikes at particular values such as 62, and the spikes are the values that can only be reached through a claim. The GBM had found the rule by which the score is updated. This is a shape effect of one factor, not an interaction, and it is something the GLM's bands cannot see because each band mixes ladder and off-ladder values.

**Evidence on the training data.** One extra parameter reduces the 4-fold cross-validated deviance from 0.455363 to 0.451617, a reduction of 0.003746. No interaction candidate comes within a quarter of that (D24). The likelihood-ratio statistic is 1,059 on one degree of freedom.

**What the fitted model says.** A policy off the ladder has 2.21 times the claim frequency of a policy in the same band on the ladder (coefficient 0.791, standard error 0.024). With the indicator in the model the band relativities also become a smooth, increasing curve, where before they were distorted by the mix:

| `BonusMalus` band | 50 | 51-54 | 55-64 | 65-79 | 80-94 | 95-99 | 100 | 101-119 | 120+ |
|---|---|---|---|---|---|---|---|---|---|
| Relativity without the indicator | 1.00 | 1.31 | 2.22 | 2.33 | 2.57 | 3.39 | 4.68 | 7.03 | 9.66 |
| Relativity with the indicator | 1.00 | 0.99 | 1.40 | 1.67 | 2.17 | 2.97 | 4.21 | 6.39 | 8.75 |

**Scope note.** CLAUDE.md's Phase 4 asks for interactions. This term is not one, but it comes from the same step (explaining the GBM) and is selected by the same criterion (cross-validation on training data). The results below report the interaction on its own, the indicator on its own and both together, so each can be judged separately.

### D24. Choosing the interaction

**Candidates.** Pairs of factors ranked by mean absolute SHAP interaction value on the training sample:

| Pair | SHAP interaction strength |
|---|---|
| `DrivAge` and `BonusMalus` | 0.082 |
| `BonusMalus` and `Region` | 0.051 |
| `BonusMalus` and `VehBrand` | 0.051 |
| `BonusMalus` and `Density` | 0.037 |
| `VehAge` and `BonusMalus` | 0.028 |
| `VehAge` and `VehBrand` | 0.023 |

Every one of the top five involves `BonusMalus`.

**Banding inside interactions.** A factor taking part in an interaction uses coarse bands, each a merge of its fine bands: driver age 18-25, 26-40, 41+; bonus-malus 50, 51-99, 100+; vehicle age 0-1, 2-12, 13+; vehicle power 4-5, 6-8, 9+. The main effects keep their fine bands. This keeps the number of extra parameters small and every cell populated. Candidates needing more than 50 extra parameters are excluded as unusable rating terms (none of the top five did).

**Selection.** Each candidate was added to the GLM and scored by 4-fold cross-validated Poisson deviance on the training data. The baseline is 0.455363.

| Interaction added | Extra parameters | CV deviance reduction | LR statistic |
|---|---|---|---|
| `BonusMalus` (3 bands) by `Region` | 42 | 0.000877 | 323 |
| `BonusMalus` (3 bands) by `VehBrand` | 20 | 0.000549 | 197 |
| `DrivAge` (3 bands) by `BonusMalus` (3 bands) | 4 | 0.000541 | 164 |
| `VehAge` (3 bands) by `BonusMalus` (3 bands) | 4 | 0.000290 | 96 |
| `BonusMalus` (3 bands) by log(`Density`) | 2 | 0.000265 | 77 |

**Chosen: `BonusMalus` by `Region`**, the largest reduction in cross-validated deviance. The same interaction is also the best one when the ladder indicator is already in the model (reduction 0.000860).

**Why the likelihood-ratio test is only supporting evidence.** Every candidate has a p-value below 1e-16. With 535,000 rows that is what happens to any real effect however small, so the test cannot rank the candidates or say whether one is worth its parameters. Cross-validated deviance measures what matters: whether the term improves predictions on policies it was not fitted to.

**A judgement call to review.** By the stated criterion the choice is `BonusMalus` by `Region`. It costs 42 parameters, and its meaning (the bonus-malus gradient is flatter in some regions, for example around Paris and the south-east) is less intuitive than driver age by bonus-malus, which has the strongest SHAP interaction, needs 4 parameters and gets 62% of the cross-validated gain. A pricing team might well prefer the smaller term. I followed the criterion and did not look at the holdout for the alternative. Note also that the driver age by bonus-malus gain falls from 0.00054 to 0.00021 once the ladder indicator is in the model: part of what looked like that interaction was the ladder effect.

### D25. The additive GBM

**Decision.** A second GBM in which every tree may split on one factor only (`interaction_constraints`, one group per feature). Its prediction is then a sum of one-factor effects on the log scale: the same structure as a GLM with no interactions, but with completely flexible shapes. It was tuned by the same 30-trial protocol (chosen: learning rate 0.1095, 11 leaves, 418 trees; CV deviance 0.450151). A test checks that the fitted model has no interactions.

**What it is for.** It splits the gap between the GLM and the GBM in two:

- GLM to additive GBM: what the GLM loses through banding and straight-line terms (shape).
- Additive GBM to GBM: what it loses by having no interactions.

The split is approximate, because the three models also differ in how they are regularised.

### D26. Results on the holdout

**Superseded as a pricing result (D30 to D32).** These are the nine-factor results. They are kept because they are how the timing problem was found: the ladder indicator closing two thirds of the gap was the result that looked too good. "How to state the result" and "What this says about the trade-off" below were written before D30 and should not be quoted. Neither GLM in the final results has an interaction.

| Model | Parameters | Poisson deviance | Deviance explained | Gini | Actual / expected |
|---|---|---|---|---|---|
| GLM | 63 | 0.45391 | 4.85% | 0.297 | 1.000 |
| GLM + interaction | 105 | 0.45339 | 4.96% | 0.298 | 1.000 |
| GLM + ladder indicator | 64 | 0.44991 | 5.69% | 0.315 | 1.001 |
| GLM + ladder indicator + interaction | 106 | 0.44940 | 5.79% | 0.316 | 1.001 |
| Additive GBM | | 0.44876 | 5.93% | 0.319 | 1.002 |
| GBM, `BonusMalus` constrained | | 0.45106 | 5.44% | 0.310 | 0.999 |
| GBM | | 0.44780 | 6.13% | 0.320 | 1.000 |

**Where the gap comes from.** The GLM-to-GBM gap is 0.00611 of deviance.

| Part of the gap | Deviance | Share | 95% bootstrap interval |
|---|---|---|---|
| Shape: GLM to additive GBM | 0.00515 | 84% | 0.00411 to 0.00624 |
| Interactions: additive GBM to GBM | 0.00096 | 16% | 0.00008 to 0.00179 |

Five sixths of what the GBM gains over the GLM is the shape of single factors, not interactions.

**How much each GLM extension closes.**

| Extension | Deviance reduction from the GLM | 95% bootstrap interval | Share of gap closed (deviance) | Share of gap closed (Gini) |
|---|---|---|---|---|
| Interaction (`BonusMalus` by `Region`) | 0.00052 | -0.00001 to 0.00101 | 8% | 7% |
| Ladder indicator | 0.00400 | 0.00313 to 0.00486 | 66% | 79% |
| Both | 0.00451 | 0.00356 to 0.00558 | 74% | 82% |

Against the right denominators: the interaction closes 54% of the interaction portion (0.00052 of 0.00096), and the ladder indicator closes 78% of the shape portion (0.00400 of 0.00515).

**How to state the result.**

- The one interaction selected from the SHAP analysis closes 8% of the gap, and its interval just includes zero, so on this holdout it cannot be claimed as a significant improvement. That is consistent with interactions being only 16% of the gap to begin with.
- One extra parameter that encodes how the bonus-malus score is updated closes two thirds of the gap. The GLM with that term (64 parameters, deviance 0.44991) is ahead of the monotone-constrained GBM (0.45106).
- With both terms the GLM reaches 0.44940. The remaining distance to the GBM is 0.00159 (interval 0.00051 to 0.00270): the GBM is still better, by about a quarter of the original gap.

**What this says about the trade-off.** The GBM's advantage here was not mainly "non-linearity and interactions" in general. Most of it was one specific piece of structure that a person can read off a SHAP dependence plot and write down as a rating factor. The GBM was useful as a way of finding it, and a pricing team that needs a GLM can keep most of the benefit.

**Pure premium (secondary).** Gini on capped losses, with the shared severity model: GLM 0.335, GLM + ladder indicator 0.348, GBM 0.348.

---

## Phase 2: the from-scratch NumPy solver

### D27. What it is, and what it is not

**Decision.** `src/irls.py` is a GLM solver written with NumPy only. It fits the two models the pipeline uses: Poisson with a log link and an offset, and Gamma with a log link and prior weights. It is **not** the pipeline's engine; statsmodels is. Its job is to show that the fitting algorithm is understood, by reproducing the statsmodels results on the pipeline's own design matrices.

**The algorithm.** With a log link, `eta = X beta + offset` and `mu = exp(eta)`, so `d mu / d eta = mu`. Each iteration:

1. Working response: `z = (eta - offset) + (y - mu) / mu`. This is the linear predictor plus the residual moved onto the linear scale. The offset is taken out because it has no coefficient to estimate.
2. Working weights: `W = w * (d mu / d eta)^2 / V(mu)`, where `w` is the prior weight and `V` is the variance function.
   - Poisson, `V(mu) = mu`: `W = w * mu`.
   - Gamma, `V(mu) = mu^2`: `W = w`. The weights do not change between iterations.
3. Weighted least squares: `beta = (X' W X)^-1 X' W z`.

Iteration stops when the deviance changes by less than 1e-10 in relative terms. Starting values are fitted means halfway between each observation and the overall mean, so that `log(mu)` is defined when `y = 0`.

**After convergence.** The covariance of the coefficients is `phi * (X' W X)^-1`. The dispersion `phi` is 1 for Poisson. For Gamma it is the Pearson chi-squared statistic over the residual degrees of freedom, `sum( w (y - mu)^2 / mu^2 ) / (n - p)`, which is the convention statsmodels uses by default.

### D28. IRLS and Newton-Raphson: when they are the same

This is on the CV, so it needs to be exact.

- **Newton-Raphson** updates with the observed information: `beta + (-H)^-1 g`, where `g` is the gradient of the log-likelihood and `H` its Hessian.
- **Fisher scoring** replaces `-H` with its expectation, the Fisher information `X' W X`.
- **IRLS is Fisher scoring** written as a weighted regression. Always.

Whether Fisher scoring equals Newton-Raphson depends on the link.

- **Poisson with a log link.** The log link is the canonical link for the Poisson family. The gradient is `X'(y - mu)` and the Hessian is `-X' diag(mu) X`, which does not contain `y`. So the observed and expected information are the same matrix, and IRLS is exactly Newton-Raphson.
- **Gamma with a log link.** The canonical link for the Gamma family is the inverse, not the log. Up to the dispersion, the log-likelihood is `sum( w (-y / mu - log mu) )`. The gradient is `X' diag(w) (y - mu) / mu`. The observed Hessian is `-X' diag(w y / mu) X`, which does contain `y`. Its expectation, using `E[y] = mu`, is `-X' diag(w) X`. IRLS uses the second, so it is Fisher scoring and not Newton-Raphson. The two take different steps away from the optimum and converge to the same estimate.

`tests/test_irls.py` checks both statements numerically: for Poisson, one IRLS step equals one hand-written Newton step to 1e-10; for Gamma, the IRLS step equals the Fisher scoring step, differs from the Newton step, and both iterations end at the same coefficients.

**The accurate CV wording** is "fitted by IRLS (equivalent to Newton-Raphson under the canonical link)", which is what the CV already says. It would be wrong to say the Gamma severity model is fitted by Newton-Raphson.

### D29. Validation against statsmodels

Seven tests in `tests/test_irls.py`. The tolerance in the tests is a relative 1e-6 with an absolute 1e-8 (a relative tolerance alone fails for a coefficient that is close to zero). What was achieved:

| | Frequency GLM | Severity GLM |
|---|---|---|
| Design matrix | 534,900 by 63 | 19,721 by 10 |
| Largest coefficient difference, absolute | 2.9e-14 | 6.6e-14 |
| Largest coefficient difference, relative | 2.6e-12 | 1.1e-12 |
| Largest standard error difference, relative | 4.4e-9 | 6.2e-14 |
| Deviance difference | 1.5e-11 (of 128,552) | 3.6e-12 |
| Dispersion | 1 (fixed) | 13.0122399303 in both |
| Iterations, NumPy and statsmodels | 7 and 7 | 7 and 9 |

The other tests use synthetic data with known coefficients, for both families: the solver matches statsmodels and recovers the true values to within four standard errors.

**On the final frequency GLM (added in Phase 5).** The test now runs on both frequency design matrices. On the final model's matrix (534,900 by 55, without `BonusMalus`): largest coefficient difference 4.0e-13 absolute and 6.0e-12 relative, largest standard error difference 9.1e-9 relative, deviance difference 2.9e-11 (of 132,754), 7 iterations for both. The severity model is unchanged.

---

## Review before Phase 5

### D30. `BonusMalus` appears to contain the claims it is used to predict

**In one line.** The evidence that `BonusMalus` was recorded after some of the claims it is used to predict is strong but unproven. The final models leave it out (D31).

**The question.** freMTPL2 does not say when `BonusMalus` was recorded. If it is the coefficient at the start of the exposure period, it is a rating factor. If it was recorded later, a claim in the data can already have moved it, and then it is partly a copy of the answer (target leakage). The ladder indicator (D23) is the term most exposed to this, because "off the claim-free ladder" means exactly "has been surcharged for a claim". Every number below comes from `src/check_bonus_malus_timing.py`, on training rows only, and is saved in `outputs/bonus_malus_timing.json`.

**How the French coefficient moves** (needed to read the tables). It is updated once a year at renewal. A claim-free year multiplies it by 0.95, truncated to two decimals, down to a floor of 0.50. An at-fault claim multiplies it by 1.25. So a driver at 50 who is surcharged goes to 62, and then, claim-free, to 58, 55, 52 and back to 50. Among those four values, the value tells you how many renewals ago the surcharge happened.

**What clean data would look like.** A past claim is a genuine predictor: it says something about the driver. In a credibility model the effect is moderate and fades slowly, a few per cent per claim-free year. It should not depend on how long the policy has been in the insurer's file, and it should be the same whether the record covers a full year or part of one.

**What the data shows.**

1. *The share Leo asked for.* 17.8% of policies with a claim are off the ladder, against 7.9% of policies without one. 92.1% of off-ladder policies have no claim in the period. On its own this fits either reading, so it does not settle the question.

2. *The effect depends on the length of the record.* The single relativity of 2.21 in D23 is an average of three very different numbers.

| Records | Frequency on the ladder | Frequency off the ladder | Ratio | GLM relativity of the indicator |
|---|---|---|---|---|
| Under half a year | 0.099 | 0.143 | 1.5 | 1.33 |
| Half a year to one year | 0.067 | 0.143 | 2.1 | 1.86 |
| Full year | 0.040 | 0.199 | 5.0 | 3.86 |

3. *In full-year records, a surcharged value is close to a record of a claim in that year.* Share of policies with at least one claim: 3.6% at 50; 3.0% at 51, 54 and 57; 31.7% at 62; 26.8% at the other values one surcharge above a claim-free value (63, 67, 71, 75, 106, 112, 118, 125). A prior claim does not make a driver nine times as likely to claim: a genuine effect of the size seen in the newest policies (under 2 times, point 4) would put the share near 7%, not 32%. If the coefficient was read after the period, a policy at 62 was surcharged at a renewal inside the period, for a claim in the year before that renewal, and part of that year lies inside the observation period. That would produce a share of this size. The drivers at 51, 54 and 57 also claim less often than the drivers at 50 (3.0% against 3.6%), although they are younger. That is the wrong way round for a rating factor, and it is consistent with claimants having been moved off those values.

4. *The value that carries the effect moves with the policy number.* This is the strongest evidence. Training policies are cut into eight equal blocks by `IDpol`. `IDpol` appears to follow the order in which policies entered the file (block 8 is 98% part-year records, as new business would be), although that is an inference. Claims per policy-year:

| `IDpol` block | 50 | 62 (surcharged at the last renewal) | 58 (one renewal later) | 55 (two later) | 52 (three later) |
|---|---|---|---|---|---|
| 1 (oldest) | 0.058 | 0.332 | 0.158 | 0.151 | 0.158 |
| 2 | 0.054 | 0.325 | 0.238 | 0.138 | 0.174 |
| 3 | 0.052 | 0.238 | 0.171 | 0.153 | 0.072 |
| 4 | 0.050 | 0.199 | 0.188 | 0.129 | 0.054 |
| 5 | 0.048 | 0.305 | 0.203 | 0.048 | 0.075 |
| 6 | 0.049 | 0.265 | 0.202 | 0.049 | 0.052 |
| 7 | 0.046 | 0.443 | 0.086 | 0.038 | 0.073 |
| 8 (newest) | 0.052 | 0.090 | 0.101 | 0.044 | 0.053 |

The table is a staircase. In the newest block no value is far above 50. One block back, only 62 is. Further back 58 joins, then 55, then 52. Each step back in the file adds one more renewal's worth of surcharge that lines up with a recorded claim. That is what you would see if the claims counted on a policy include the claims that produced its coefficient, and it is not what a driver characteristic does: a claim three renewals ago cannot matter threefold for long-standing policies and not at all for recent ones. The cells are small (between 8 and 62 claims in the lower-right half), so single cells are uncertain; the pattern across 32 cells is not.

The newest block is 98% part-year records, so one objection is that the table only shows that part-year records behave differently. Among part-year records alone, the newest block still shows no excess at 62: 0.045 claims per policy-year against 0.054 at 50, on 9 claims. Every older block shows 62 at 2.4 to 5.3 times the frequency at 50 (0.385 against 0.085 in the oldest). So the difference between the newest block and the rest survives. The renewal-by-renewal steps for 58, 55 and 52 are noisier in part-year records alone, and I would not quote them on that basis.

Block 8 is the nearest thing to a clean estimate of the genuine effect of a past claim: 1.7 to 1.9 times at 62 and 58, on about 20 claims each, and nothing at 55 and 52.

**What is not established.** The mechanism. A year-end snapshot explains point 3 but not the staircase; the staircase suggests that either claims from more than one year are attached to older policies, or policies were observed in different years and the coefficient was read later. The documentation (the CASdatasets reference page) describes `BonusMalus` only as "between 50 and 350: <100 means bonus, >100 means malus in France". It says the policies were "observed mostly on one year" within a period of "approximately 2011-2013", which leaves room for different policies to have been observed in different years. It does not say when the coefficient was read. I found no published account of the problem. One public benchmark repository (TabArena data-foundry, issue 51) mentions that its version of freMTPL2 blanks `BonusMalus` in some `IDpol` blocks, without saying why. That fits with someone else having seen a block-dependent problem, but it is not evidence of the mechanism. The evidence here is circumstantial and comes from the data alone. It is strong enough that the opposite claim, that the 2.2 ratio reflects prior history and not the same claims, cannot be written honestly.

**Why this matters beyond the indicator.** `BonusMalus` is in every model. Cross-validated deviance on the training data, with and without it (the GBM keeps its tuned parameters and is not re-tuned):

| | Null | GLM | GBM | GLM-to-GBM gap |
|---|---|---|---|---|
| Nine factors | 0.47925 | 0.45536 | 0.44875 | 0.00661 |
| Without `BonusMalus` | 0.47925 | 0.47016 | 0.46981 | 0.00035 |

- `BonusMalus` is 62% of what the GLM explains and 69% of what the GBM explains.
- 95% of the GLM-to-GBM gap exists only when `BonusMalus` is in the models. The GBM uses the raw value and can learn the sawtooth; the GLM's bands blur it. On the holdout the gap is 0.0061 unconstrained and 0.0029 with the monotone constraint, which stops the GBM using the sawtooth.
- The "66% of the gap closed by one parameter" (D26) is the GLM being taught the same pattern. It is not evidence of a better pricing model.
- D17 dropped the monotone constraint because the sawtooth made it expensive. If the sawtooth is partly an artefact, that reasoning no longer holds.
- The level of every Gini and deviance figure is flattered to an unknown degree, the baseline GLM's included.

Not affected: the data cleaning, the split, the metrics, the severity model (driver age only) and the NumPy solver validation.

**I should have caught this in Phase 4.** One parameter closing two thirds of the gap is a result that "looks too good", and CLAUDE.md says to suspect leakage first. D23 described the indicator as marking a past claim without testing whether the claim was past.

**Options.**

- **A. Keep the nine factors, drop the indicator, report both views (recommended).** The final GLM is the baseline. `BonusMalus` stays because every published freMTPL2 comparison uses it and part of its effect is genuine, but the write-up leads with this finding and adds a second, pre-declared comparison without `BonusMalus` (GLM and a GBM tuned for it, scored once on the holdout). The honest statement is then a range: the GBM's advantage is 0.006 with the suspect factor and close to nothing without it. About 45 minutes of compute.
- **B. Remove `BonusMalus` from every model** and redo Phases 1, 3 and 4. Cleanest, but it discards a real rating factor on circumstantial evidence, and the likely result is that the GBM does not beat the GLM. Three to four hours of compute.
- **C. Keep the indicator with a one-line caveat**, as first proposed. Not recommended: the indicator is the term that leans hardest on the suspect pattern, and an actuary who knows the French system would ask this question first.

**Leo's decision (17 November 2026).** A variant of option A. The final models leave `BonusMalus` out of both the GLM and the GBM (D31). The nine-factor results stay in the results table as a labelled comparison: the analysis that exposed the timing problem, not a pricing improvement. Neither GLM has an interaction. The monotone constraint stays closed (D17).

---

## Phase 5: the final model

### D31. The final model, specified before it was scored

This entry was written and committed before the final models were scored on the holdout. Nothing in it was changed after the scores were seen. The results are in D32.

**Why `BonusMalus` is dropped rather than kept with a caveat.** A factor that may contain the outcome cannot be defended in a price, and no caveat fixes the numbers it produces. Dropping it costs a real part of the signal (D30: block 8 suggests a genuine prior-claim effect of under 2 times), but what is left is a model whose inputs are all known when the policy is written. A pricing actuary would rather have a weaker model that is clean than a stronger one that may be scoring itself.

**The final GLM.** Poisson, log link, offset `log(Exposure)`, fitted by statsmodels without a penalty. Terms: `DrivAgeBand` (ten bands), `VehAgeBand`, `VehPowerBand`, `LogDensity`, `VehBrand`, `VehGas`, `Region`. 55 parameters (the nine-factor GLM had 63; `BonusMalusBand` took 8). The bands and reference levels are those of D11. They were chosen on training data with `BonusMalus` in the model and are not chosen again, because choosing them again would add a selection step for one model and not the other. `Area` stays out (D11). No interaction, no ladder indicator.

**The final GBM.** LightGBM, Poisson objective, `init_score = log(Exposure x training base rate)` (D16). Eight raw factors: `Area`, `VehPower`, `VehAge`, `DrivAge`, `VehBrand`, `VehGas`, `Density`, `Region`, with brand and region ordered by training frequency (D18). No monotone constraint. Tuned under the protocol of D19: the same 30 random parameter draws, 4-fold cross-validation on the training data, early stopping after 50 rounds without improvement. Chosen (`outputs/gbm_tuning_final.json`):

| Learning rate | Leaves | Minimum data in leaf | L2 penalty | Trees | Cross-validated deviance |
|---|---|---|---|---|---|
| 0.1264 | 17 | 1,791 | 13.3636 | 92 | 0.469645 |

All 30 trials fall between 0.469645 and 0.470040. Compared with the nine-factor GBM (26 leaves, at least 84 policies per leaf, 166 trees), the search prefers a smaller, more heavily regularised model: with `BonusMalus` gone there is less structure to find.

**Shared parts.** The severity model is the Gamma GLM on driver age (D12), unchanged, because it never used `BonusMalus`. Each frequency model gets one scalar rebalancing factor fitted on training data (D13); the pure premium is rebalanced to capped training losses and the large-loss load is reported separately (D5).

**Training cross-validation, before scoring** (same four folds; the GLM figure is from `outputs/bonus_malus_timing.json`):

| Null | Final GLM | Final GBM | GLM minus GBM |
|---|---|---|---|
| 0.479249 | 0.470162 | 0.469645 | 0.000517 |

The cross-validated gap is 8% of the nine-factor gap (0.006614). The spread across the 30 tuning trials (0.000395) is almost as large as the gap, so whether the GBM is ahead on the holdout is an open question.

**What will be reported, whatever it shows.**

- Primary: holdout Poisson deviance of each model, and the GLM-minus-GBM difference with a paired bootstrap interval (1,000 resamples of holdout policies, seed 42; D21). If the interval includes zero, the write-up says the GBM does not beat the GLM on these eight factors.
- Also: Gini (D8), actual over expected overall, by rating factor and by length of record, and the double-lift chart of the pair (D9).
- Secondary: pure premium Gini and actual over expected, on capped losses and on all losses.
- The share-of-gap-closed ratio is not computed for the final pair, because no term is added to the final GLM.

**What the holdout had already been used for.** Before this decision, seven nine-factor models were scored on the same holdout: the GLM, the GLM with the interaction, with the ladder indicator, with both, the additive GBM, the constrained GBM and the GBM. Two choices were then made with those scores in view. Dropping the interaction (Leo's decision of 13 November) followed its holdout interval. The `BonusMalus` investigation rests on training data alone (D30), but what prompted it was a holdout result that looked too good. The final models have not been scored before. Their specification was not chosen on the holdout, but the route to it passed through holdout results, so the final figures may be mildly optimistic. With 133,726 policies and differences of a few thousandths, I expect that effect to be small, but it cannot be measured.

### D32. Results of the final models on the holdout

Scored once, after D31 was committed (`python src/run_final.py`; the first run stopped with an error before any holdout metric was computed, and the second run is the one reported). Holdout: 133,726 policies, 70,827 policy-years, 5,215 claims.

| Frequency model | Parameters | Poisson deviance | Deviance explained | Gini | Actual / expected |
|---|---|---|---|---|---|
| Null (training frequency) | 1 | 0.47703 | 0 | 0 | |
| Final GLM | 55 | 0.46866 | 1.75% | 0.179 | 0.998 |
| Final GBM | | 0.46833 | 1.82% | 0.183 | 0.998 |

**The GBM does not beat the GLM.** The GLM's deviance exceeds the GBM's by 0.00033, with a 95% paired bootstrap interval of -0.00019 to 0.00085. The interval includes zero. On the eight rating factors other than `BonusMalus`, this holdout cannot tell the two models apart. The cross-validated gap on the training data was 0.00052, so the holdout agrees with what training data predicted.

**Overfitting.** Training against holdout deviance: GLM 0.46954 and 0.46866; GBM 0.46611 and 0.46833. The GBM's holdout score is close to its cross-validated score (0.46965). The holdout is a little easier than the training data for both models (its null deviance is 0.47703 against 0.47925 in cross-validation), which is why both score better there.

**Rebalancing factors.** Frequency: GLM 1.0000 (balance property), GBM 0.99996. Pure premium on capped losses: GLM 1.0017, GBM 1.0022. Large-loss load 1.1917. None needed investigating.

**The double-lift chart** (`plots/double_lift_final_glm_vs_gbm.png`; the dashboard draws the same table):

| Group | GBM / GLM | Actual | GLM | GBM |
|---|---|---|---|---|
| 1 | 0.49 to 0.86 | 0.0757 | 0.0862 | 0.0699 |
| 2 | 0.86 to 0.91 | 0.0671 | 0.0786 | 0.0698 |
| 5 | 0.97 to 1.00 | 0.0754 | 0.0728 | 0.0716 |
| 6 | 1.00 to 1.02 | 0.0760 | 0.0717 | 0.0725 |
| 9 | 1.10 to 1.17 | 0.0758 | 0.0696 | 0.0785 |
| 10 | 1.17 to 1.98 | 0.0772 | 0.0686 | 0.0864 |

The actual frequency barely moves across the ten groups (0.067 to 0.077), while the two models' predictions fan out in opposite directions. The GBM is closer to the actual in six groups and the GLM in four, and in both end groups the actual lies between the two. So the GBM's departures from the GLM point the right way on average but are too large. This is what a small, not significant deviance difference looks like on a double-lift chart. Compare D20, where the actual line followed the GBM to 0.124 in the top group.

**What leaving out `BonusMalus` costs.** Measured against the nine-factor models on the same holdout: 0.01475 of deviance for the GLM (interval 0.01287 to 0.01676) and 0.02053 for the GBM (0.01818 to 0.02289). Deviance explained falls from 4.85% to 1.75% for the GLM. This is an upper bound on the price of the decision, because part of what `BonusMalus` contributed may be the leak. The final GLM's actual over expected by bonus-malus band runs from 0.76 at 50 to 4.10 at 120 and above: a real insurer, which knows each driver's coefficient at the start of the policy, would never price without it. The final model is what this dataset supports cleanly, not what an insurer would use.

**Driver age now carries what bonus-malus used to.** The final GLM's driver-age relativities (against 41 to 50) are 3.84 at 18 to 19, 2.60 at 20 to 21, 1.95 at 22 to 23, 1.42 at 24 to 25, 1.04 at 26 to 30 and 0.90 at 31 to 40. The dip at 26 to 40 in the nine-factor model (D11) has gone, as D11 explained it would: young drivers can only have high bonus-malus scores, so with that factor present the model had already charged them through it.

**Record length.** Actual over expected on the holdout by length of record, for both final models: about 0.80 for full-year records, 0.99 for records of half a year to a year, 1.43 for records under half a year. For the nine-factor models the same figures are about 0.86, 0.96 and 1.32. On the training data the observed frequency is 0.054 claims per policy-year for full-year records, 0.079 for half to one year and 0.112 for under half a year (and 0.299 for records of 0.05 years or less). Claim frequency is not proportional to exposure in this data, so the log-exposure offset misprices short and long records in opposite directions. Both models inherit this equally, because neither is allowed to use exposure as a rating factor (D3). That `BonusMalus` reduced the misfit is one more sign that it carries information about the period being observed.

**Pure premium (secondary).** Capped losses: Gini 0.199 (GLM) and 0.205 (GBM); actual over expected 0.974 and 0.973. All losses: Gini 0.207 and 0.212; without the policy with the largest holdout loss, 0.183 and 0.188. Actual over expected on all losses after the large-loss load: 0.834 and 0.833 (the holdout has few large claims; D14).

**What it means for GLM against GBM.** With `BonusMalus` the GBM's lead was 0.0061 and well clear of zero. Without it the lead is 0.0003 and within noise. Almost all of the GBM's apparent advantage came through the fine structure of one factor that may contain the outcome. On the remaining factors, a GLM with bands chosen by cross-validation captures what the data support. This is one dataset and one holdout, and the holdout had been used before (D31), so the right statement is narrow: on freMTPL2 without `BonusMalus`, a tuned GBM did not beat a well-specified GLM.

### D33. Phase 5 practical choices

- **The dashboard reads one precomputed file**, `outputs/model_comparison.json`, written by `src/summarise_models.py`. It fits no model and imports only `streamlit`, `altair`, `pandas`, `json` and `pathlib`; a test parses its imports and another runs the app headless on the saved results. At Leo's request it has three parts: the headline table for the final pair, their double-lift chart, and a short section on the nine-factor comparison and why it was set aside. It has no sliders, so the "commercial scenario overlay" in CLAUDE.md does not arise, and the Lorenz curve, one-way and dislocation charts of the legacy app are not rebuilt.
- **`requirements.txt`** now pins only what the app imports: `streamlit` 1.53.0, `altair` 6.0.0, `pandas` 2.3.3 and `numpy` 2.3.4. This resolves the old `numpy<2.0.0` pin. The legacy app's requirements are kept in `legacy/requirements.txt`.
- **The Python version.** Streamlit Community Cloud ignores `runtime.txt`; the version is chosen in the app's Advanced settings. The pinned packages have wheels for Python 3.11 to 3.14. The pipeline was run on 3.13.9, so 3.13 is the version to select.
- **The legacy pipeline** moved to `legacy/` with its own `config.py`, `requirements.txt` and charts. `src/config.py` keeps only the dataset IDs and the seed. The audit scripts now import the legacy code from `legacy/`.
- **The Excel engine is a new file**, `artifacts/rating_engine_final_glm.xlsx`, built by `src/build_rating_engine.py`. The legacy workbook `artifacts/rating_engine.xlsx` is not touched. Its hand-built `Sample_Policy` and `Sensitivity_OneWay` sheets use what-if data tables, array formulas and a chart, which openpyxl cannot carry from one workbook to another without damage. Copying them was therefore not an option, and they stay in the legacy workbook. The new workbook prices from raw rating factors with Excel formulas only (`LOOKUP` for bands, `INDEX`/`MATCH` on the rating table) and has a one-policy calculator. After it was recalculated in Excel, its Reconcile sheet matches Python for 100 of 100 holdout policies, the largest relative difference being 1.6e-15. A one-way sensitivity sheet can be rebuilt by hand on the Calculator sheet if wanted.
