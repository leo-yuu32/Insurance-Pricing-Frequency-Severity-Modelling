# Phase summaries

Written for Leo's review. Each phase starts with the plan that was followed (plan mode was skipped for this run, at Leo's request), then what changed, the key numbers, and anything surprising. The reasoning behind each choice is in `docs/DECISIONS.md` (entries D1 to D29).

**State of the repo.** Five stacked local branches, nothing pushed, `main` untouched:

`phase-0-audit` -> `phase-1-foundations` -> `phase-3-gbm` -> `phase-4-shap-interactions` -> `phase-2-numpy-glm` (the tip, which contains everything).

42 tests pass (`python -m pytest`). The legacy pipeline and the dashboard are unchanged and still run.

**To reproduce everything** (PowerShell, from the repo root, after `python -m pip install -r requirements-dev.txt`):

```powershell
python src\run_glm.py            # baseline GLMs; first run downloads the data and builds the split (about 5 minutes)
python src\tune_gbm.py           # about 35 minutes
python src\tune_gbm.py monotone  # about 75 minutes
python src\tune_gbm.py additive  # one to two hours
python src\run_gbm.py            # about 2 minutes
python src\run_interactions.py   # about 20 minutes
python src\summarise_models.py   # writes outputs\model_comparison.json
python -m pytest
```

---

## Headline

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

## Not done

- **Phase 5:** dashboard rebuild, README, `INTERVIEW_NOTES.md`, moving the legacy pipeline to `legacy/`.
- **Excel regeneration.** It is for the final GLM only, and which GLM is final is your call (see below). The hand-built `Sample_Policy` and `Sensitivity_OneWay` sheets are untouched.

I stopped here because Phase 5 and the Excel workbook both depend on the decisions below.

## Decisions waiting for you

1. **Which GLM is the final one?** Baseline, with the ladder indicator, or with the indicator and an interaction.
2. **Which interaction, if any?** `BonusMalus` by `Region` (the criterion's choice) or driver age by bonus-malus (smaller and more intuitive).
3. **Is the grouped split acceptable?** It departs from "only group if material".
4. **`CLAUDE.md`** has the approved edits applied and is still untracked. Two wordings go beyond what you approved: the legacy move is deferred to Phase 5 so the deployed app keeps working, and modelling dependencies go in `requirements-dev.txt`. `requirements.txt` is untouched.
5. **The CV.** Supported now: statsmodels GLMs on nine factors; GLM against a tuned LightGBM on a grouped holdout with a bootstrap interval; a double-lift chart; SHAP; the NumPy solver validated on the real design matrix. Still to come off: the Gini 0.43 to 0.48 line and "isotonic calibration".
