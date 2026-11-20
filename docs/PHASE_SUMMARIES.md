# Phase summaries

Written for Leo's review. Each phase starts with the plan that was followed (plan mode was skipped for this run, at Leo's request), then what changed, the key numbers, and anything surprising. The reasoning behind each choice is in `docs/DECISIONS.md` (entries D1 to D33). The newest phase is first.

## Phase 5: final model, dashboard, Excel engine and write-up (17 to 20 November 2026)

**Plan** (from your decisions of 17 November):

1. Retune the GBM on the eight factors other than `BonusMalus`, on training cross-validation only, with the same 30 parameter draws as before.
2. Write the full specification of both final models into `DECISIONS.md` (D31) and commit it before any holdout scoring.
3. Score the final pair once on the holdout, with a bootstrap interval for the gap.
4. Build the results file and the three-part dashboard, the Excel engine for the final GLM (new file), and move the legacy pipeline to `legacy/`.
5. Write the README, `DECISIONS.md` (D32, D33, and notes on the entries D30 overturns), `INTERVIEW_NOTES.md` and `CV_SUMMARY.md`.

**What changed.**

- New code: `src/run_final.py`, `build_rating_engine.py` and `dashboard.py`.
- Rewritten: `src/summarise_models.py` and `src/config.py`.
- `gbm.py` and `tune_gbm.py` accept a feature list (`tune_gbm.py final`).
- `legacy/` holds the old pipeline, its config, requirements and charts.
- `requirements.txt` is pinned for the app; `requirements-dev.txt` includes it.
- New workbook `artifacts/rating_engine_final_glm.xlsx`.
- 12 new tests (60 in total, all passing).
- Commit `f954934` holds the specification and `2f9c9da` the scores, in that order.

**Key numbers** (holdout, 133,726 policies):

| | Final GLM (55 parameters) | Final GBM |
|---|---|---|
| Poisson deviance | 0.46866 | 0.46833 |
| Deviance explained | 1.75% | 1.82% |
| Gini | 0.179 | 0.183 |
| Actual / expected | 0.998 | 0.998 |

- **The gap is not significant.** GLM minus GBM is 0.00033, with a 95% interval of -0.00019 to 0.00085. Without `BonusMalus` the GBM does not beat the GLM. Cross-validation on training data had predicted 0.00052.
- **Tuning chose a heavily regularised GBM:** learning rate 0.1264, 17 leaves, at least 1,791 policies per leaf, L2 13.36, 92 trees. All 30 trials lie within 0.0004 of each other.
- **Leaving out `BonusMalus` costs a lot:** 0.0147 of deviance for the GLM and 0.0205 for the GBM. That is an upper bound, since part of it may be the leak.
- **Excel:** the new workbook reproduces Python for 100 of 100 holdout policies, the largest relative difference being 1.6e-15, checked after recalculating in Excel itself.
- **NumPy solver:** on the final 534,900 by 55 matrix it matches statsmodels to 6.0e-12 for coefficients and 9.1e-9 for standard errors.

**Surprising.**

- **The final double-lift chart looks nothing like the nine-factor one.** The actual frequency is nearly flat across the ten groups (0.067 to 0.077) while the two models fan out in opposite directions, and in both end groups the actual lies between them.
- **Driver age takes over.** Without `BonusMalus` the driver-age relativities become steep and monotone (3.84 at 18 to 19 against 41 to 50). The dip at 26 to 40 disappears, as D11 predicted.
- **Mis-calibration by record length is worse without `BonusMalus`.** Actual over expected is 0.80 for full-year records and 1.43 for records under half a year, against 0.86 and 1.32 with it. So the factor carries information about the observed period, one more sign of the timing problem.

**Where I departed from the letter of the instructions, and why.**

- **Dashboard.** It has only the three parts you asked for. `CLAUDE.md` also asks for SHAP plots, dislocation between models, scenario sliders and rebuilt legacy tabs. Your instruction to keep it simple took precedence. Consequence: the CV line about dislocation analysis and Lorenz curves no longer matches any app (`CV_SUMMARY.md`, section 8).
- **Excel.** The engine is a new file. The hand-built `Sample_Policy` and `Sensitivity_OneWay` sheets stay, untouched, in `artifacts/rating_engine.xlsx`, because they use what-if data tables, array formulas and a chart that openpyxl cannot copy without damage. That workbook stays in `artifacts/`, not `legacy/`, because it has your uncommitted edits.
- **GLM bands.** The final GLM keeps the bands chosen with `BonusMalus` present. Re-choosing them would have added a selection step for one model only (D31).
- **Additive GBM.** It was not rebuilt without `BonusMalus`: with no significant gap there is nothing to split.
- **Earlier entries.** D17's "is this leakage?" argument now carries a correction, and D20 and D26 are marked as superseded. The original text is kept.
- **The first `run_final.py` run** stopped with an error before any holdout metric was computed. It was fixed (a missing column) and run again. The fix is in the scoring commit.
- **`requirements.txt`** had your uncommitted one-line change (a trailing newline). The new file includes it.

**Before this goes live.** I have not pushed anything or touched `main`. To deploy:

1. Review the branches.
2. Merge `phase-5-final-model` (which contains every earlier branch) into `main` and push.
3. In the Streamlit Community Cloud settings, select Python 3.13 and check that the app's main file is `src/dashboard.py`.

The app then reads `outputs/model_comparison.json` and fits nothing.

**Still open.**

- `CLAUDE.md` is untracked, and its "Current state" section describes the project before the rebuild.
- The repository URL: the git remote is `github.com/idk-what-to-put-here-32/...`, while the old README gave `github.com/leo-yuu32/...`. `CV_SUMMARY.md` uses the remote.

---

## Review of the five decisions before Phase 5 (13 November 2026): decided on 17 November, see Phase 5 above

The stop rule applies. The leakage check asked for in decision 3 came back the wrong way, which undoes decision 1 and changes how the headline numbers below can be read. Phase 5 and the CV summary have not been started. The evidence and the options are in `docs/DECISIONS.md`, D30.

| Decision | Verdict | Reason |
|---|---|---|
| 1. Final GLM = baseline + ladder indicator, no interaction | **Disagree** on the indicator. Agree on leaving the interaction out and on not scoring driver age by bonus-malus on the holdout. | The indicator leans on a pattern that looks like target leakage (decision 3). One correction to the reasoning on the interaction: dropping it because of its holdout result is itself a holdout-based choice. It is acceptable because it is the conservative direction and all four GLMs stay in the table. |
| 2. Grouped split, no monotone constraint, CLAUDE.md wordings | Agree on the split and the wordings. **The monotone constraint is reopened.** | D17 dropped the constraint because the bonus-malus sawtooth made it expensive. The sawtooth is now suspect. |
| 3. Leakage check on the indicator | **Disagree with the conclusion it presupposes.** The check was the right thing to ask for. | The evidence does not show that the 2.2 ratio reflects prior history. It points the other way, and the question applies to `BonusMalus` in every model, not only to the indicator. |
| 4. Phase 5 | Not started | It depends on which GLM is final. |
| 5. CV summary | Not started | Same. |

**What the check found** (training rows only, `src/check_bonus_malus_timing.py`):

- The 2.2 relativity is an average of 1.3 in records under half a year, 1.9 in records of half a year to a year, and 3.9 in full-year records.
- In full-year records, 32% of policies at 62 (one surcharge above 50) have a claim in the year, against 3.6% at 50. A prior claim does not make a driver nine times as likely to claim; a coefficient recorded after the claim does.
- The bonus-malus value that carries the excess moves with the policy number: no value in the newest eighth of policies, only 62 in the next, then 62 and 58, then 55 and 52 as well in the oldest. A driver characteristic does not behave like that. A coefficient that already reflects the recorded claims does. The newest block still shows no excess on part-year records alone, so this is not just a difference in record length (but only 9 claims sit in that cell).
- In cross-validation on the training data, the GLM-to-GBM gap is 0.0066 with `BonusMalus` and 0.0003 without it.

**What this changes.** The 66% "gap closed by one parameter" is not a pricing improvement. Most of the GBM's advantage over the GLM runs through the suspect factor. The level of every Gini and deviance figure is flattered to an unknown degree. The mechanism is not proven; the evidence is circumstantial but consistent.

**My recommendation (option A in D30).** Final GLM = baseline, no indicator, no interaction. Keep the nine-factor results, lead the write-up with this finding, and add one pre-declared comparison without `BonusMalus`, scored once on the holdout. About 45 minutes of compute.

---

**State of the repo.** Seven stacked local branches, nothing pushed, `main` untouched:

`phase-0-audit` -> `phase-1-foundations` -> `phase-3-gbm` -> `phase-4-shap-interactions` -> `phase-2-numpy-glm` -> `bonus-malus-timing-check` -> `phase-5-final-model` (the tip, which contains everything).

60 tests pass (`python -m pytest`). The legacy pipeline is in `legacy/` and still imports.

**To reproduce everything:** see "Reproducing the results" in `README.md`.

---

## Headline of Phases 1 to 4 (nine factors, now set aside)

**These are the nine-factor results.** They are correct as computed and stay in the results as the analysis that exposed the `BonusMalus` timing problem (D30). They are not pricing results. The final results are in the Phase 5 section.

Holdout of 133,726 policies, scored once. Poisson deviance per policy-year, lower is better.

| Model | Deviance | Gini | Share of GLM-to-GBM gap closed |
|---|---|---|---|
| GLM (statsmodels, 63 parameters) | 0.45391 | 0.297 | |
| GLM + one interaction (`BonusMalus` by `Region`) | 0.45339 | 0.298 | 8% |
| GLM + bonus-malus ladder indicator (1 parameter) | 0.44991 | 0.315 | 66% |
| GLM + both | 0.44940 | 0.316 | 74% |
| Additive GBM (no interactions) | 0.44876 | 0.319 | |
| GBM | 0.44780 | 0.320 | |

- The GBM beats the GLM by 0.0061 (95% bootstrap interval 0.0049 to 0.0073).
- 84% of that gap is the shape of single factors and 16% is interactions.
- The interaction chosen from the SHAP analysis closes 8% of the gap, and its interval just includes zero.
- One extra GLM parameter, found from the GBM's SHAP dependence plot, closes 66%.

---

## Phase 1: shared foundations

**Plan.** Settle the data conventions and log them. Build one data module (cleaning and a new split), one metrics module, one design-matrix builder, and the statsmodels baseline. Back each GLM design choice with cross-validation on training data.

**What changed.** New modules `src/data.py`, `metrics.py`, `design.py`, `glm.py`, `run_glm.py`, `compare_glm_designs.py`; 27 tests; `docs/DECISIONS.md`; `requirements-dev.txt`.

**Key numbers.**

- 668,626 policies after excluding the anomalous block; 534,900 train, 133,726 holdout.
- Baseline GLM on the holdout: deviance 0.45391 against 0.47703 for the null model (4.85% explained), frequency Gini 0.297, actual over expected 0.9997.
- Pure premium rebalancing factor 1.0017 (the legacy pipeline needed 0.816).

**Your decision 2 (excluding the block): I agree.** Excluding it and counting claims from the severity records give the same portfolio frequency to within 0.2% (0.0739 and 0.0738). Exclusion is right under either reading of what the block is; the alternative is right under only one. Cost: 1.34% of exposure. One correction: the block ends at `IDpol` 24,951, not 24,500. The 248 policies in between all have claims too, so I cut where the pattern stops.

**Surprising.**

- 38% of policies share all nine rating factors with another policy, and a random split would put a twin of a third of the holdout into training. I grouped the split by profile. A direct check found no leak, so this is caution, not a measured need.
- `BonusMalus` is banded, not linear as I proposed in the audit. Linear costs 10% of everything the model explains.
- Severity: every factor except driver age makes out-of-fold predictions worse. The severity GLM uses driver age only.
- The EUR 4.08 million claim now falls in the training set. It is half of the large-loss load (1.19), which is reported as a flat load and kept out of the model.
- `BonusMalus` and `Density`, which the legacy model lacked, account for most of what the GLM explains.

---

## Phase 3: LightGBM against the GLM

**Plan.** Poisson objective with exposure as an offset. Same nine factors as the GLM, raw. Tune by cross-validation on training data. Decide the monotone constraint on training data. Compare on the holdout with a bootstrap interval and a double-lift chart.

**What changed.** `src/gbm.py`, `tune_gbm.py`, `run_gbm.py`, `compare_gbm_constraints.py`; 6 tests covering both `init_score` traps; `plots/double_lift_glm_vs_gbm.png`.

**Key numbers.**

- GBM 0.44780 against GLM 0.45391 on the holdout.
- Tuning barely matters: 30 trials span 0.0006 of deviance.
- Double lift: in the tenth of the book where the GBM is highest relative to the GLM, the actual frequency is 0.124; the GLM predicts 0.075 and the GBM 0.122.

**Surprising: the monotone constraint was dropped.** CLAUDE.md suggested constraining `BonusMalus` to be increasing. In cross-validation on training data the constraint costs 0.0030, nearly as much as the constrained GBM gains over the GLM. The reason is that the bonus-malus effect is a sawtooth: scores on the claim-free ladder (50, 51, 54, 57, 60, 64 ... 100) have low frequencies, and scores off it, which can only be reached after a claim (62 = 50 x 1.25), run two to three times higher within the same band. The unconstrained model is the primary GBM; the constrained one is tuned and reported (0.45106).

**Also.** Brand and region are fed to the GBM as integers ordered by training frequency, because SHAP interaction values fail on LightGBM's native categorical splits.

**A bug I made and caught.** One version of `run_gbm.py` still applied the constraint to the "unconstrained" model. Its holdout score came out identical to the constrained one to six digits, which gave it away.

---

## Phase 4: SHAP, and what goes back into the GLM

**Plan.** SHAP values and interaction values on training samples. Turn the top pairs into GLM interaction terms with coarse bands. Choose by cross-validated deviance on training data, with the likelihood-ratio test as supporting evidence. Refit and score on the holdout. Report the share of the gap closed.

**What changed.** `src/run_interactions.py`, `summarise_models.py`; the ladder indicator and coarse bands in `design.py`; SHAP plots in `plots/`; `outputs/model_comparison.json` as the single results table.

**Key numbers.** See the headline table.

**Two things I did beyond "one interaction". Please check you want both.**

1. **The ladder indicator.** The SHAP dependence plot for `BonusMalus` shows the sawtooth. I added one GLM term for "score below 100 and off the claim-free ladder" (relativity 2.21). It is a main effect, not an interaction, so it is outside the letter of Phase 4, but it comes from the same step and is by far the largest finding. The results report the interaction alone, the indicator alone and both, so you can drop it cleanly.
2. **The additive GBM**, a stretch item. It was cheap to run alongside and it explains the headline: interactions are only 16% of the gap, which is why one interaction closes so little.

**A judgement call for you.** The criterion picks `BonusMalus` by `Region` (42 parameters). Driver age by bonus-malus has the strongest SHAP interaction, needs 4 parameters and gets 62% of the cross-validated gain. It is the easier one to defend to an actuary. I followed the criterion and did not score the alternative on the holdout.

---

## Phase 2 (stretch): the NumPy solver

**Plan.** Move the solver into the repo as a module with offset, prior weights, Poisson and Gamma. Test it against statsmodels on synthetic data and on the pipeline's real design matrices. State the Newton-Raphson equivalence exactly.

**What changed.** `src/irls.py`; 7 tests in `tests/test_irls.py`.

**Key numbers.** On the 534,900 by 63 frequency matrix the coefficients agree with statsmodels to 2.6e-12 relative and the standard errors to 4.4e-9. Both converge in 7 iterations. Two tests show numerically that the Poisson step is a Newton-Raphson step and that the Gamma log-link step is Fisher scoring.

---

## Decisions taken (17 November 2026)

1. **The final model** leaves out `BonusMalus` in both the GLM and the GBM. Neither GLM has an interaction. Driver age by bonus-malus was never scored on the holdout.
2. **Accepted:** the grouped split and both `CLAUDE.md` wordings. The monotone constraint stays closed.
3. **The `BonusMalus` timing evidence** is described as strong but unproven (D30).
4. **Phase 5 and `CV_SUMMARY.md`** are done (see the top of this file).
