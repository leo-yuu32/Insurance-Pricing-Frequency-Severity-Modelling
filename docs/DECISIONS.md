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
