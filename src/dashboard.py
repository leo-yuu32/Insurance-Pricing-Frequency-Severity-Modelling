"""Streamlit dashboard: the final GLM and GBM compared on the holdout.

It reads outputs/model_comparison.json (written by src/summarise_models.py) and nothing else. It
fits no model and does not import statsmodels, lightgbm or shap.

    streamlit run src/dashboard.py
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RESULTS_PATH = Path(__file__).resolve().parent.parent / "outputs" / "model_comparison.json"
SERIES_COLOURS = {"Actual": "#222222", "GLM": "#1f77b4", "GBM": "#d62728"}


def load_results(path=RESULTS_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics_table(models):
    """One row per model, with the holdout metrics shown on the page."""
    return pd.DataFrame([{
        "Model": m["label"],
        "Poisson deviance": m["poisson_deviance"],
        "Deviance explained": m["deviance_explained"],
        "Gini": m["gini"],
        "Actual / expected": m["a_over_e"],
    } for m in models.values()]).set_index("Model")


def format_table(table):
    return table.style.format({"Poisson deviance": "{:.5f}", "Deviance explained": "{:.2%}", "Gini": "{:.3f}", "Actual / expected": "{:.3f}"})


def interval_sentence(diff, better="GBM", worse="GLM"):
    """Plain statement of a deviance difference and whether its interval excludes zero."""
    est, low, high = diff["deviance_reduction"], diff["bootstrap_95_low"], diff["bootstrap_95_high"]
    text = f"The {better}'s deviance is lower than the {worse}'s by {est:.5f} (95% bootstrap interval {low:.5f} to {high:.5f}). "
    if low > 0:
        return text + f"Since the interval excludes zero, the {better} scores better on the holdout."
    if high < 0:
        return text + f"Since the interval lies below zero, the {worse} scores better on the holdout."
    return text + f"The interval includes zero, so the {better} does not beat the {worse} on these factors."


def double_lift_long(rows):
    """The double-lift table in long form: one row per group and series."""
    d = pd.DataFrame(rows)
    out = d.melt(id_vars=["bin"], value_vars=["actual_rate", "rate_glm", "rate_gbm"], var_name="series", value_name="rate")
    out["series"] = out["series"].map({"actual_rate": "Actual", "rate_glm": "GLM", "rate_gbm": "GBM"})
    return out


def double_lift_chart(rows):
    data = double_lift_long(rows)
    return alt.Chart(data).mark_line(point=True).encode(
        x=alt.X("bin:O", title="Group (1 = GBM lowest relative to GLM, 10 = highest)"),
        y=alt.Y("rate:Q", title="Claims per policy-year", scale=alt.Scale(zero=False), axis=alt.Axis(format=".3f")),
        color=alt.Color("series:N", title=None, scale=alt.Scale(domain=list(SERIES_COLOURS), range=list(SERIES_COLOURS.values()))),
        tooltip=[alt.Tooltip("series:N", title="Series"), alt.Tooltip("bin:O", title="Group"), alt.Tooltip("rate:Q", title="Frequency", format=".4f")],
    ).properties(height=360)


def main():
    st.set_page_config(page_title="MTPL frequency: GLM and GBM", layout="centered")
    r = load_results()
    final, nine = r["final"], r["with_bonus_malus"]

    st.title("Motor claim frequency: GLM and GBM")
    st.markdown(
        f"We compare two models of claim frequency on French motor third-party liability policies (freMTPL2). Both take "
        f"exposure as an offset, and both are fitted on {r['train']['policies']:,} training policies and scored once on "
        f"{r['holdout']['policies']:,} holdout policies ({r['holdout']['claims']:,} claims)."
    )

    st.header("Final models: eight rating factors, without BonusMalus")
    st.dataframe(format_table(metrics_table(final["models"])), width="stretch")
    st.markdown(interval_sentence(final["glm_minus_gbm"]))

    st.subheader("Double-lift chart, GLM against GBM")
    st.altair_chart(double_lift_chart(final["double_lift"]), width="stretch")
    st.markdown(
        "We sort the holdout policies by the ratio of the GBM's prediction to the GLM's and cut them into ten groups of "
        "equal exposure, and for each group we plot the actual claim frequency and each model's average prediction. Where "
        "the two models disagree, the better model is the one whose line stays closer to the actual line."
    )

    st.header("With BonusMalus: the comparison that was set aside")
    st.dataframe(format_table(metrics_table(nine["models"])), width="stretch")
    ladder = nine["gap_closed"].get("glm_ladder", {}).get("share_of_gap_closed_deviance")
    st.markdown(
        interval_sentence(nine["differences"]["glm_to_gbm"])
        + (f" One extra GLM term, flagging bonus-malus scores that can only be reached through a claim, closes {ladder:.0%} of that gap." if ladder else "")
    )
    st.markdown(
        "This result led us to investigate the factor itself. Three checks on the training data suggest that `BonusMalus` "
        "was recorded after some of the claims it is used to predict. Its effect is several times stronger in full-year "
        "records than in short ones. In full-year records, nearly a third of the policies whose score was raised at the last "
        "renewal have a claim in the same period. And which scores carry the excess shifts with the policy number. Since the "
        "dataset does not say when the score was recorded, the evidence is strong but unproven. We therefore keep these "
        "figures as the analysis that exposed the problem, not as a pricing result, and the final models leave the factor out."
    )

    st.caption(
        "Poisson deviance is per policy-year (lower is better). Gini orders policies by predicted frequency and is 0 for a "
        "model with no discrimination. Each model gets one scalar rebalancing factor fitted on training data. Note that the "
        "holdout was used for several comparisons and one selection decision before the final models were fixed, and so "
        "their figures may be mildly optimistic. The GLMs are fitted with statsmodels and independently reproduced by a "
        "NumPy IRLS solver."
    )


if __name__ == "__main__":
    main()
