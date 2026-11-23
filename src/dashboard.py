"""Streamlit dashboard: GLM vs GBM for motor claim frequency.

It reads the small precomputed files in artifacts/dashboard/ (written by src/build_dashboard_data.py)
and nothing else. It fits no model and does not import statsmodels, lightgbm or shap.

    streamlit run src/dashboard.py
"""

import json
import textwrap
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "artifacts" / "dashboard"
REPO_URL = "https://github.com/leo-yuu32/Insurance-Pricing-Frequency-Severity-Modelling"
LINKEDIN_URL = "https://www.linkedin.com/in/leo-yuu"
SPEC_COMMIT_URL = REPO_URL + "/commit/f954934"
DISCLOSURE = "The rebuild of this project was developed with an AI coding assistant (Claude Code) under my direction."

# The same colours on every chart, checked for colour-blind separation: GLM blue, GBM orange, actual dark grey.
GLM_BLUE, GBM_ORANGE, ACTUAL_GREY, MUTED_GREY = "#0072B2", "#E07000", "#3A3A3A", "#C9C9C9"
# The site's accent (section labels and the theme's primary colour), dark enough for small text: 5.2:1 on white.
ACCENT = "#C2410C"
SERIES = {"Actual": ACTUAL_GREY, "GLM": GLM_BLUE, "GBM": GBM_ORANGE}
TITLE_WIDTH, SUBTITLE_WIDTH = 42, 60  # characters per line, so that titles fit a phone screen
RECORD_LENGTHS = ["under half a year", "half to one year", "full year"]


# ---------------------------------------------------------------------------------------------
# Data and wording
# ---------------------------------------------------------------------------------------------

REQUIRED_FILES = ["models", "driver_age", "shap", "score_rates", "price_ratio", "lift", "double_lift", "lorenz",
                  "relativities", "record_length", "timing_checks"]


def data_version(path=DATA_DIR):
    """Name, size and modification time of every data file. Passing this to the cached loader means
    that a rebuilt or added file is read again instead of being served from the cache."""
    return tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(Path(path).glob("*.json")))


@st.cache_data
def _read_files(path, version):
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(path).glob("*.json"))}


def load_data(path=DATA_DIR):
    """Every precomputed file, keyed by file name without the extension."""
    return _read_files(str(path), data_version(path))


def model(data, key):
    return next(r for r in data["models"]["rows"] if r["key"] == key)


def verdict(low, high):
    """Words for a reduction in prediction error whose 95% interval is (low, high)."""
    if low > 0:
        return "clearly and significantly better"
    if high < 0:
        return "clearly and significantly worse"
    return "within the margin of error"


def times(x):
    """A ratio as words, rounded to a whole number ("three times")."""
    words = {2: "twice", 3: "three times", 4: "four times", 5: "five times", 6: "six times", 7: "seven times",
             8: "eight times", 9: "nine times", 10: "ten times"}
    n = round(x)
    return words.get(n, f"{n} times")


def share_of_lead_lost(data):
    """The share of the GBM's lead with the no-claims score that is gone without it."""
    return 1.0 - model(data, "gbm_final")["reduction"] / model(data, "gbm")["reduction"]


def pct(x):
    """A percentage to two decimals, or three when it would otherwise round to zero."""
    return f"{x:.3f}%" if 0 < abs(x) < 0.005 else f"{x:.2f}%"


def models_table(rows):
    """Holdout metrics, one row per model, indexed by model and formatted for display. The last two
    columns are the change in prediction error against the GLM with the same rating factors."""
    out = []
    for r in rows:
        if r["reference"]:
            change, interval = "reference", ""
        else:
            word = "lower" if r["pct"] >= 0 else "higher"
            change, interval = f"{pct(abs(r['pct']))} {word}", f"{pct(r['low_pct'])} to {pct(r['high_pct'])}"
        out.append({
            "Model": r["label"],
            "Prediction error": f"{r['poisson_deviance']:.5f}",
            "Deviance explained": f"{100 * r['deviance_explained']:.2f}%",
            "Gini": f"{r['gini']:.3f}",
            "A / E": f"{r['a_over_e']:.3f}",
            "Against the GLM": change,
            "95% interval": interval,
        })
    return pd.DataFrame(out).set_index("Model")


def gbm_closer_groups(rows):
    """Number of double-lift groups in which the GBM's prediction is closer to the actual rate."""
    return sum(abs(r["rate_gbm"] - r["actual_rate"]) < abs(r["rate_glm"] - r["actual_rate"]) for r in rows)


def staircase_newest_max_ratio(data):
    """In the newest block of policies, the highest claim rate among 62, 58, 55 and 52 over the rate at 50."""
    rows = [r for r in data["timing_checks"]["check3_staircase"] if r["block"].startswith("8 ")]
    base = next(r["rate"] for r in rows if r["value"] == "50")
    return max(r["rate"] for r in rows if r["value"] != "50") / base


# ---------------------------------------------------------------------------------------------
# Charts (Altair only)
# ---------------------------------------------------------------------------------------------

def chart_title(text, subtitle=None):
    params = {"text": textwrap.wrap(text, TITLE_WIDTH), "anchor": "start", "fontSize": 15, "offset": 12}
    if subtitle:
        params.update(subtitle=textwrap.wrap(subtitle, SUBTITLE_WIDTH), subtitleFontSize=12, subtitleColor="#555555")
    return alt.TitleParams(**params)


def series_colour(domain=("Actual", "GLM", "GBM")):
    return alt.Color("series:N", title=None, scale=alt.Scale(domain=list(domain), range=[SERIES[s] for s in domain]),
                     legend=alt.Legend(orient="top", symbolOpacity=1))


def driver_age_chart(rows):
    d = pd.DataFrame(rows).rename(columns={"actual": "Actual", "glm": "GLM", "gbm": "GBM"})
    long = d.melt(id_vars=["age"], value_vars=["Actual", "GLM", "GBM"], var_name="series", value_name="rate")
    base = alt.Chart(long).encode(
        x=alt.X("age:Q", title="Driver age", scale=alt.Scale(domain=[18, 85], nice=False)),
        y=alt.Y("rate:Q", title="Claims per driver per year", axis=alt.Axis(format=".2f")),
        color=series_colour(),
        tooltip=[alt.Tooltip("series:N", title="Series"), alt.Tooltip("age:Q", title="Age"),
                 alt.Tooltip("rate:Q", title="Claims per driver per year", format=".3f")])
    actual = base.transform_filter(alt.datum.series == "Actual").mark_circle(size=30, opacity=0.75)
    glm = base.transform_filter(alt.datum.series == "GLM").mark_line(interpolate="step", strokeWidth=2.5)
    gbm = base.transform_filter(alt.datum.series == "GBM").mark_line(strokeWidth=2.5)
    return alt.layer(actual, glm, gbm).properties(
        height=440, title=chart_title("The GLM's prediction moves in steps, the GBM's changes gradually",
                                      "Holdout drivers aged 18 to 85"))


def shap_chart(rows):
    d = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    d["highlight"] = d["factor"] == "No-claims score"
    ratio = d["mean_abs_shap"].iloc[0] / d["mean_abs_shap"].iloc[1]
    return alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("mean_abs_shap:Q", title="Average effect (mean |SHAP|)"),
        y=alt.Y("factor:N", sort=list(d["factor"]), title=None),
        color=alt.when(alt.datum.highlight).then(alt.value(GBM_ORANGE)).otherwise(alt.value(MUTED_GREY)),
        tooltip=[alt.Tooltip("factor:N", title="Factor"), alt.Tooltip("mean_abs_shap:Q", title="Mean |SHAP|", format=".3f")],
    ).properties(height=420, title=chart_title(
        f"The GBM relied on the no-claims score {times(ratio)} as much as on any other factor",
        "Nine-factor GBM, 10,000 training policies"))


def score_chart(rows):
    d = pd.DataFrame(rows)
    d["x0"], d["x1"] = d["score"] - 0.4, d["score"] + 0.4
    d["highlight"] = d["score"].isin([50, 62])
    d["label"] = d["score"].astype(str) + ": " + d["rate"].map(lambda v: f"{v:.3f}")
    ratio = d.loc[d["score"] == 62, "rate"].iloc[0] / d.loc[d["score"] == 50, "rate"].iloc[0]
    x_scale, x_axis = alt.Scale(domain=[49, 101], nice=False), alt.Axis(values=[50, 60, 70, 80, 90, 100])
    y = alt.Y("rate:Q", title="Claims per driver per year")
    bars = alt.Chart(d).mark_bar().encode(
        x=alt.X("x0:Q", title="No-claims score", scale=x_scale, axis=x_axis),
        x2="x1:Q",
        y=y,
        y2=alt.datum(0),
        color=alt.when(alt.datum.highlight).then(alt.value(ACTUAL_GREY)).otherwise(alt.value(MUTED_GREY)),
        tooltip=[alt.Tooltip("score:Q", title="Score"), alt.Tooltip("policies:Q", title="Full-year records", format=","),
                 alt.Tooltip("rate:Q", title="Claims per driver per year", format=".3f")])
    labels = alt.Chart(d[d["highlight"]]).mark_text(dy=-8, fontSize=12, color=ACTUAL_GREY, align="left", dx=-6).encode(
        x=alt.X("score:Q", title="No-claims score", scale=x_scale, axis=x_axis), y=y, text="label:N")
    return alt.layer(bars, labels).properties(height=400, title=chart_title(
        f"Drivers at 62 claim {times(ratio)} as often as drivers at 50",
        "Full-year records in the training data, scores with at least 300 such records"))


def with_without_chart(data):
    with_score, without = model(data, "gbm"), model(data, "gbm_final")
    d = pd.DataFrame([
        {"comparison": "With the|no-claims score", "pct": with_score["pct"], "low": with_score["low_pct"], "high": with_score["high_pct"]},
        {"comparison": "Without the|no-claims score", "pct": without["pct"], "low": without["low_pct"], "high": without["high_pct"]},
    ])
    d["label"] = d["pct"].map(lambda v: f"{v:.2f}%")
    y = alt.Y("comparison:N", sort=list(d["comparison"]), title=None,
              axis=alt.Axis(labelExpr="split(datum.label, '|')", labelFontSize=12))
    scale = alt.Scale(domain=[min(-0.2, d["low"].min() - 0.1), d["high"].max() + 0.2])
    title = "GBM's error, % below the GLM's"
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color="#888888").encode(x=alt.X("x:Q", title=title, scale=scale))
    interval = alt.Chart(d).mark_rule(strokeWidth=3, color=GBM_ORANGE).encode(
        x=alt.X("low:Q", title=title, scale=scale), x2="high:Q", y=y)
    dots = alt.Chart(d).mark_circle(size=140, color=GBM_ORANGE, opacity=1).encode(
        x=alt.X("pct:Q", title=title, scale=scale), y=y,
        tooltip=[alt.Tooltip("pct:Q", title="% lower", format=".2f"), alt.Tooltip("low:Q", title="95% interval from", format=".2f"),
                 alt.Tooltip("high:Q", title="95% interval to", format=".2f")])
    labels = alt.Chart(d).mark_text(dy=-14, fontSize=12, color=ACTUAL_GREY).encode(
        x=alt.X("pct:Q", title=title, scale=scale), y=y, text="label:N")
    return alt.layer(zero, interval, dots, labels).properties(height=300, title=chart_title(
        f"The GBM's advantage falls from {with_score['pct']:.2f}% to {without['pct']:.2f}%",
        "Holdout, with 95% paired bootstrap intervals"))


def price_ratio_chart(price):
    d = pd.DataFrame(price["bins"])
    d["share_pct"] = 100 * d["share"]
    d["group"] = d["differs"].map({True: "More than 10% apart", False: "Within 10%"})
    return alt.Chart(d).mark_bar(binSpacing=1).encode(
        x=alt.X("low:Q", bin="binned", title="GBM price divided by GLM price",
                axis=alt.Axis(values=[0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4], format=".1f")),
        x2="high:Q",
        y=alt.Y("share_pct:Q", title="Share of drivers (%)"),
        color=alt.Color("group:N", title=None, scale=alt.Scale(domain=["Within 10%", "More than 10% apart"],
                                                               range=[MUTED_GREY, ACTUAL_GREY]), legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("low:Q", title="Ratio from", format=".2f"), alt.Tooltip("high:Q", title="Ratio to", format=".2f"),
                 alt.Tooltip("share_pct:Q", title="Share of drivers (%)", format=".2f")],
    ).properties(height=380, title=chart_title(
        f"For {100 * price['share_differing']:.0f}% of drivers the two prices differ by more than 10%",
        "Holdout policies, weighted by time insured"))


def lift_chart(rows):
    d = pd.DataFrame(rows).rename(columns={"actual_rate": "Actual", "rate_glm": "GLM", "rate_gbm": "GBM"})
    long = d.melt(id_vars=["bin"], value_vars=["Actual", "GLM", "GBM"], var_name="series", value_name="rate")
    base = alt.Chart(long).encode(
        x=alt.X("bin:O", title="Group (lowest predicted risk to highest)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("rate:Q", title="Claims per driver per year", axis=alt.Axis(format=".2f")),
        color=series_colour(),
        tooltip=[alt.Tooltip("series:N", title="Series"), alt.Tooltip("bin:O", title="Group"),
                 alt.Tooltip("rate:Q", title="Claims per driver per year", format=".4f")])
    # The GLM is drawn wide and pale under the GBM, so that both stay visible where they overlap.
    glm = base.transform_filter(alt.datum.series == "GLM").mark_line(strokeWidth=7, opacity=0.45)
    gbm = base.transform_filter(alt.datum.series == "GBM").mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36))
    actual = base.transform_filter(alt.datum.series == "Actual").mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=50))
    spreads = [times(d[c].iloc[-1] / d[c].iloc[0]) for c in ["Actual", "GLM", "GBM"]]
    title = f"The highest-risk group claims {spreads[0]} as often as the lowest"
    if len(set(spreads)) == 1:
        title += ", as both models predict"
    return alt.layer(glm, gbm, actual).properties(height=420, title=chart_title(
        title, "Holdout, final models, ten groups of equal exposure"))


def double_lift_chart(rows):
    d = pd.DataFrame(rows).rename(columns={"actual_rate": "Actual", "rate_glm": "GLM", "rate_gbm": "GBM"})
    long = d.melt(id_vars=["bin"], value_vars=["Actual", "GLM", "GBM"], var_name="series", value_name="rate")
    n = gbm_closer_groups(rows)
    return alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=50), strokeWidth=2.5).encode(
        x=alt.X("bin:O", title="Group (GBM against GLM, low to high)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("rate:Q", title="Claims per driver per year", scale=alt.Scale(zero=False), axis=alt.Axis(format=".3f")),
        color=series_colour(),
        tooltip=[alt.Tooltip("series:N", title="Series"), alt.Tooltip("bin:O", title="Group"),
                 alt.Tooltip("rate:Q", title="Claims per driver per year", format=".4f")],
    ).properties(height=440, title=chart_title(
        f"The GBM is closer to the actual rate in {n} groups of 10, the GLM in {len(rows) - n}",
        "Final models on the holdout, ten groups of equal exposure"))


def lorenz_chart(lorenz):
    parts = []
    for key, name in [("glm_final", "GLM"), ("gbm_final", "GBM")]:
        parts.append(pd.DataFrame({"x": lorenz[key]["x"], "y": lorenz[key]["y"], "series": name}))
    d = pd.concat(parts)
    x = alt.X("x:Q", title="Share of exposure (lowest risk first)", axis=alt.Axis(format="%"))
    y = alt.Y("y:Q", title="Share of claims", axis=alt.Axis(format="%"))
    diagonal = alt.Chart(pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})).mark_line(color="#999999", strokeWidth=1).encode(x=x, y=y)
    curves = alt.Chart(d).mark_line(strokeWidth=2.5).encode(
        x=x,
        y=y,
        color=series_colour(("GLM", "GBM")),
        tooltip=[alt.Tooltip("series:N", title="Model"), alt.Tooltip("x:Q", title="Exposure", format=".0%"),
                 alt.Tooltip("y:Q", title="Claims", format=".1%")])
    g, b = lorenz["glm_final"]["gini"], lorenz["gbm_final"]["gini"]
    return alt.layer(diagonal, curves).properties(height=440, title=chart_title(
        f"The two curves almost coincide (Gini {g:.3f} and {b:.3f})", "Holdout, final GLM and GBM"))


def relativity_chart(factor):
    d = pd.DataFrame(factor["levels"])
    d["label"] = d["level"] + d["reference"].map({True: " (reference)", False: ""})
    lo, hi = min(d["relativity"].min(), 1.0), max(d["relativity"].max(), 1.0)
    pad = 0.05 * (hi - lo)
    scale, title = alt.Scale(domain=[lo - pad, hi + pad], nice=False), "Relativity (reference level = 1)"
    bars = alt.Chart(d).mark_bar(color=GLM_BLUE).encode(
        x=alt.X("relativity:Q", title=title, scale=scale),
        x2=alt.datum(1.0),
        y=alt.Y("label:N", sort=list(d["label"]), title=None),
        tooltip=[alt.Tooltip("level:N", title="Level"), alt.Tooltip("relativity:Q", title="Relativity", format=".3f")])
    one = alt.Chart(pd.DataFrame({"relativity": [1.0]})).mark_rule(color="#888888").encode(
        x=alt.X("relativity:Q", title=title, scale=scale))
    return alt.layer(bars, one).properties(height=170 + 26 * len(d), title=chart_title(
        f"{factor['name']}: relativities of the final GLM", "Each level against the level with the most exposure"))


def density_chart(density):
    d = pd.DataFrame(density["curve"])
    x = alt.X("density:Q", title="Density, people per km² (log scale)", scale=alt.Scale(type="log"))
    y = alt.Y("relativity:Q", title="Relativity", scale=alt.Scale(zero=False))
    line = alt.Chart(d).mark_line(color=GLM_BLUE, strokeWidth=2.5).encode(
        x=x, y=y,
        tooltip=[alt.Tooltip("density:Q", title="Density", format=",.0f"), alt.Tooltip("relativity:Q", title="Relativity", format=".3f")])
    ref = alt.Chart(pd.DataFrame({"density": [density["reference_density"]], "relativity": [1.0]})).mark_circle(
        size=70, color=GLM_BLUE).encode(x=x, y=y)
    return alt.layer(line, ref).properties(height=400, title=chart_title(
        f"Each tenfold rise in density multiplies the claim rate by {density['per_tenfold']:.2f}",
        f"Relative to the median training density, {density['reference_density']:,.0f} per km²"))


def record_length_chart(rows):
    d = pd.DataFrame(rows)
    long = d.melt(id_vars=["record_length"], value_vars=["a_over_e_glm", "a_over_e_gbm"], var_name="series", value_name="a_over_e")
    long["series"] = long["series"].map({"a_over_e_glm": "GLM", "a_over_e_gbm": "GBM"})
    labels = {"under half a year": "Under half|a year", "half to one year": "Half a year|to a year", "full year": "Full year"}
    long["record"] = long["record_length"].map(labels)
    y = alt.Y("a_over_e:Q", title="Actual claims over predicted")
    bars = alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("record:N", sort=[labels[k] for k in RECORD_LENGTHS], title="Length of the policy record",
                axis=alt.Axis(labelAngle=0, labelExpr="split(datum.label, '|')")),
        xOffset=alt.XOffset("series:N", sort=["GLM", "GBM"]),
        y=y,
        color=series_colour(("GLM", "GBM")),
        tooltip=[alt.Tooltip("series:N", title="Model"), alt.Tooltip("record_length:N", title="Record"),
                 alt.Tooltip("a_over_e:Q", title="Actual / expected", format=".2f")])
    one = alt.Chart(pd.DataFrame({"a_over_e": [1.0]})).mark_rule(color="#888888").encode(y=y)
    r = {row["record_length"]: row for row in rows}
    return alt.layer(bars, one).properties(height=400, title=chart_title(
        f"Actual over predicted is {r['under half a year']['a_over_e_glm']:.2f} for short records and "
        f"{r['full year']['a_over_e_glm']:.2f} for full years", "Holdout, final models (figures for the GLM)"))


# ---------------------------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------------------------

def show(chart):
    st.altair_chart(chart, width="stretch")


def section_label(text):
    """A small orange label, placed once directly above the section it introduces."""
    st.markdown(f'<p style="font-size:1.05rem; font-weight:700; color:{ACCENT}; margin:0 0 -0.6rem 0;">{text}</p>',
                unsafe_allow_html=True)


def header(data):
    """Title, subtitle, byline and the summary, above the tabs."""
    st.title("GLM vs GBM")
    st.markdown("#### Can machine learning price car insurance better than the industry-standard model?")
    st.markdown(f"by Leo Yu, Mathematics, Imperial College London · [LinkedIn]({LINKEDIN_URL}) · [GitHub]({REPO_URL})")
    policies = data["models"]["holdout"]["policies"] + data["models"]["train"]["policies"]
    with st.container(border=True):
        st.markdown(
            f"In this article, we compare the industry-standard model for pricing car insurance, a generalised linear "
            f"model (GLM), with a machine-learning model, a gradient-boosted model (GBM). Both predict how often a driver "
            f"will make a claim, and we fit and test them on {policies:,} French car insurance policies.")
        st.markdown(
            "We find that the GBM is at first clearly better, but that most of its advantage comes from a no-claims score "
            "which appears to have been recorded after some of the claims it is used to predict. Without the score, the "
            "GBM's advantage is within the margin of error, and the GLM is fully transparent.")


def data_and_cleaning(data):
    n = data["models"]["train"]["policies"] + data["models"]["holdout"]["policies"]
    st.markdown(
        f"We use freMTPL2, a public dataset of French motor third-party liability policies from the CASdatasets "
        f"collection, downloaded from OpenML. After cleaning it holds {n:,} policies with their claim counts and claim "
        f"amounts.")
    st.markdown(
        "We exclude the policies with the lowest policy numbers (up to 24,951), since every one of them has a claim "
        "count but almost none has a claim amount, and so they are not a random sample of the book. Claim counts are "
        "taken from the claim records and capped at 4 per policy, and record lengths are capped at one year.")
    st.markdown(
        "Individual claims are capped at EUR 181,271, the 99.9th percentile of training claims, and the cost above "
        "the cap is spread over all policies as one flat load (1.192).")


def split_and_validation(data):
    n_train, n_holdout = data["models"]["train"]["policies"], data["models"]["holdout"]["policies"]
    st.markdown(
        f"We split the policies once into {n_train:,} for fitting and {n_holdout:,} for testing (the holdout). We cut "
        f"the policies into five folds, stratified on whether a policy has a claim and grouped so that policies with "
        f"identical rating factors stay together, and one fold is the holdout. Banding, tuning and the choice of model "
        f"terms use 4-fold cross-validation on the other four folds.")
    st.markdown(
        "The GBM was tuned by a 30-trial random search with early stopping. The intervals are paired bootstrap "
        "intervals from 1,000 resamples of holdout policies, in which both models are scored on the same resample.")


def article(data):
    glm9, gbm9, glm8, gbm8 = (model(data, k) for k in ["glm", "gbm", "glm_final", "gbm_final"])
    n_train, n_holdout = data["models"]["train"]["policies"], data["models"]["holdout"]["policies"]

    section_label("Introduction")
    st.subheader("The two ways to predict car insurance claims")
    st.markdown(
        "Both models predict how often a driver will claim, in claims per driver per year, from the same rating factors "
        "such as driver age, vehicle power and region. The GLM (generalised linear model) is the industry standard. It "
        "multiplies together one factor per characteristic, each read from a table, and so a characteristic such as age "
        "is split into bands with one value per band.")
    st.markdown(
        "The GBM (gradient-boosted model) is a machine-learning method, and it learns the shape of each relationship "
        "from the data instead.")
    show(driver_age_chart(data["driver_age"]["rows"]))
    st.caption("Each dot is the actual claim rate of holdout drivers of that age, and each line is a model's average "
               "prediction for the same drivers.")
    with st.expander("How the two models work"):
        st.markdown(
            "The GLM assumes that the logarithm of a driver's expected claim rate is a sum of one term per rating factor. "
            "Equivalently, the rate is a base rate multiplied by one relativity per factor, and so the fitted model is a "
            "set of rating tables. We place driver ages into ten bands, and every driver in a band gets the same age "
            "relativity, which is why the GLM's line moves in steps.")
        st.markdown(
            "The GBM is a sum of 92 small decision trees, each fitted to correct the errors of the trees before it. It "
            "uses the raw age, and so it can change its prediction wherever the data suggest, which gives the smoother line.")
        st.markdown(
            f"Both models allow for the length of each policy record, so that a driver observed for six months is "
            f"expected to have half the claims of a driver observed for a full year. We fit both on {n_train:,} policies "
            f"and test them on {n_holdout:,} other policies that neither model saw during fitting (the holdout). Ages "
            f"above 85 are left out of the chart, since each has fewer than 61 policy-years in the holdout.")

    st.divider()
    section_label("Analysis")
    st.subheader("At first, the GBM was clearly better, and it relied most on the no-claims score")
    st.markdown(
        "In our first comparison both models used all nine rating factors in the data, including the no-claims score. "
        "This is France's bonus-malus coefficient: 50 is the maximum discount, 100 is the starting level, and a score "
        "above 100 is a surcharge after claims.")
    st.markdown(
        f"The GBM was {verdict(gbm9['low_pct'], gbm9['high_pct'])}, with a prediction error {gbm9['pct']:.2f}% lower "
        f"than the GLM's. Prediction error (lower is better) measures how far the predicted numbers of claims are from "
        f"the actual ones.")
    st.caption(
        f"Prediction error is the Poisson deviance per policy-year on the holdout: {glm9['poisson_deviance']:.5f} for the "
        f"GLM and {gbm9['poisson_deviance']:.5f} for the GBM. The 95% paired bootstrap interval for the reduction is "
        f"{gbm9['low_pct']:.2f}% to {gbm9['high_pct']:.2f}% of the GLM's error, and so it excludes zero.")
    show(shap_chart(data["shap"]["rows"]))
    st.caption("Each bar is the average size of one factor's effect on the GBM's prediction, and so a longer bar means "
               "the GBM relied on that factor more.")
    with st.expander("What a SHAP value is, and what it showed"):
        st.markdown(
            f"A SHAP value splits one driver's prediction into a contribution from each rating factor, and the "
            f"contributions add up exactly to the prediction on the log scale. We average the absolute contributions over "
            f"{data['shap']['rows_used']:,} training policies to measure how much the GBM relies on each factor. Note that "
            f"this measures what the model uses, not what causes claims.")
        st.markdown(
            "The GBM's curve for the no-claims score was not smooth. It had spikes at scores that can only be reached "
            "through a claim, such as 62, and when we gave the GLM one extra term flagging those scores, this single "
            "parameter closed two thirds of the gap between the models. A result that good from one parameter led us to "
            "investigate the score itself.")

    st.divider()
    st.subheader("The no-claims score appears to contain information from after the claims")
    st.markdown(
        "The dataset does not say when the no-claims score was recorded. If it was recorded at the start of each policy, "
        "it describes the driver's past and is a fair rating factor. If it was recorded later, a claim in the data may "
        "already have raised it, which is like checking a driver's no-claims score after their accident and using it to "
        "predict that accident.")
    st.markdown(
        "This is called leakage, since information from after the event leaks into the prediction. We find that the "
        "evidence for it here is strong but unproven, and so we describe the leakage as probable.")
    scores = {r["score"]: r for r in data["score_rates"]["rows"]}
    ratio = scores[62]["rate"] / scores[50]["rate"]
    show(score_chart(data["score_rates"]["rows"]))
    st.caption("Each bar is the claim rate of drivers with that score, among full-year policy records in the training "
               "data. A driver at 50 who makes a claim moves to 62 at the next renewal.")
    st.markdown(
        f"A single past claim would not be expected to make a driver {times(ratio)} as likely to claim again, and so "
        f"the high rate at 62 suggests that many of these claims are the very claims that moved the score.")
    checks = data["timing_checks"]
    rel = {r["record_length"]: r["glm_relativity"] for r in checks["check1_record_length"]}
    share = {r["group"].split(" ")[0]: r["share_with_claim"] for r in checks["check2_full_year_share"]}
    with st.expander("The three timing checks"):
        st.markdown("All three checks use the training data only.")
        st.markdown(
            f"First, the score's effect depends on the length of the record. For scores that can only be reached through "
            f"a claim, the GLM's relativity is {rel['full year']:.2f} in full-year records, {rel['half to one year']:.2f} "
            f"in records of half a year to a year and {rel['under half a year']:.2f} in records under half a year. A fact "
            f"about the driver should have the same effect whatever the length of the record, whereas a score read at "
            f"the end of the period would matter more in longer records.")
        st.markdown(
            f"Second, in full-year records {share['62']:.1%} of drivers at 62 have a claim in the same year, against "
            f"{share['50']:.1%} at 50. If a past claim roughly doubled a driver's risk, which is about what the newest "
            f"policies show, the share at 62 would be near {2 * share['50']:.0%}.")
        st.markdown(
            f"Third, the score that carries the excess moves with the policy number. Among the oldest policies, drivers "
            f"at 58, 55 and 52 (one, two and three renewals after a surcharge) also claim far more often than drivers at "
            f"50, whereas among the newest policies no score has more than {staircase_newest_max_ratio(data):.1f} times "
            f"the rate at 50. This is what we would see if the claims attached to older policies include the claims that "
            f"produced their score.")
        st.markdown(
            "None of these checks proves the mechanism, and the dataset's documentation does not settle it. Note that a "
            "real insurer knows each driver's score at the start of the policy and would use it, and so the problem lies "
            "with this dataset, not with bonus-malus as a rating factor.")

    st.divider()
    st.subheader("Without it, the GBM's advantage all but disappears")
    st.markdown(
        f"We then removed the no-claims score from both models, retuned the GBM on the training data and scored each "
        f"model once on the holdout. We committed the specification of both models before scoring them "
        f"([commit f954934]({SPEC_COMMIT_URL})).")
    st.markdown(
        f"The GBM's prediction error is now only {gbm8['pct']:.2f}% lower than the GLM's, which is "
        f"{verdict(gbm8['low_pct'], gbm8['high_pct'])}. On the remaining eight factors the difference is therefore small "
        f"and not distinguishable from zero.")
    show(with_without_chart(data))
    st.caption("Each dot is how much lower the GBM's prediction error is than the GLM's, and the line through it is the "
               "95% interval. An interval that crosses zero means the difference is within the margin of error.")
    st.caption(
        f"Without the score, prediction error (Poisson deviance per policy-year) is {glm8['poisson_deviance']:.5f} for the "
        f"GLM and {gbm8['poisson_deviance']:.5f} for the GBM. The reduction is {gbm8['reduction']:.5f}, with a 95% "
        f"paired bootstrap interval of {gbm8['reduction_low']:.5f} to {gbm8['reduction_high']:.5f}.")
    with st.expander("Why the advantage disappears"):
        lost = share_of_lead_lost(data)
        st.markdown(
            f"{'Over 90%' if lost > 0.9 else f'{lost:.0%}'} of the GBM's lead in the first comparison goes once the score "
            f"is removed. The GBM used the raw score and could learn its spikes, whereas the GLM's bands blurred them, and "
            f"so most of what the GBM had found was the pattern we now believe to be leakage.")
        calibration = max(abs(1 - glm8["a_over_e"]), abs(1 - gbm8["a_over_e"]))
        st.markdown(
            f"Note that removing the score also makes both models less accurate, since part of its effect is genuine, "
            f"and the GLM's prediction error rises from {glm9['poisson_deviance']:.5f} to {glm8['poisson_deviance']:.5f}. "
            f"The two final models rank drivers almost equally well (Gini {glm8['gini']:.3f} for the GLM and "
            f"{gbm8['gini']:.3f} for the GBM), and both predict the total number of claims to within "
            f"{100 * calibration:.1f}%.")

    st.divider()
    section_label("Conclusion")
    st.subheader("Without the no-claims score, both models agree on who is high risk")
    price = data["price_ratio"]
    st.markdown(
        "We sort the holdout policies by the average of the two models' predictions and cut them into ten groups of "
        "equal exposure, from the lowest predicted risk to the highest. For each group we plot the actual claim rate and "
        "each model's average prediction.")
    show(lift_chart(data["lift"]["rows"]))
    st.caption("If the three lines move together, the two models sort drivers the same way and both match what actually "
               "happened.")
    st.markdown(
        f"However, individual prices still differ, and for {100 * price['share_differing']:.0f}% of drivers the two "
        f"models' prices are more than 10% apart.")
    with st.expander("How far individual prices differ"):
        st.markdown(
            "For each holdout policy we divide the GBM's price by the GLM's, where the price is the expected cost of "
            "claims per year.")
        show(price_ratio_chart(price))
        st.caption("Each bar is the share of drivers, weighted by time insured, with that ratio of GBM price to GLM "
                   "price. The darker bars are more than 10% apart.")
        st.markdown(
            f"Of the holdout exposure, {price['share_higher']:.1%} would pay more than 10% more under the GBM and "
            f"{price['share_lower']:.1%} more than 10% less. The two models agree on average and rank drivers about "
            f"equally well, and so their disagreements on individual drivers largely cancel out.")
        st.markdown(
            f"In practice, moving from one model to the other would change many customers' prices, which pricing teams "
            f"call dislocation, and so a switch needs a reason beyond accuracy. The bars at the two ends of the chart "
            f"include the {price['share_beyond_axis']:.1%} of exposure beyond the axis.")

    st.divider()
    st.subheader("Method and limitations")
    record = {r["record_length"]: r for r in data["record_length"]["rows"]}
    st.markdown(
        f"We note three limitations. First, the evidence on the no-claims score is strong but unproven, since the "
        f"dataset does not say when the score was recorded. Second, both models assume that claims grow in proportion "
        f"to the length of the record, which does not hold in this data, and so they over-predict full-year records "
        f"(actual over expected {record['full year']['a_over_e_glm']:.2f} for the GLM) and under-predict records under "
        f"half a year ({record['under half a year']['a_over_e_glm']:.2f}). Third, the holdout was used to compare several "
        f"earlier models, and one choice (dropping an interaction term) was made with those results in view, and so the "
        f"final figures may be mildly optimistic, although the final models were committed before they were scored.")
    with st.expander("Method in detail"):
        data_and_cleaning(data)
        split_and_validation(data)
        st.markdown(f"{DISCLOSURE} The code and the full method are on [GitHub]({REPO_URL}).")

    st.divider()
    st.markdown(
        f"On this data, and without the no-claims score, the GBM's advantage over the GLM is "
        f"{verdict(gbm8['low_pct'], gbm8['high_pct'])}. The GLM therefore matches the GBM's accuracy while staying fully "
        f"transparent, and the score's apparent value most likely came from information recorded after the claims.")
    st.caption(f"[GitHub repository]({REPO_URL}) · [LinkedIn]({LINKEDIN_URL})")


def quantitative_analysis(data):
    holdout = data["models"]["holdout"]

    section_label("Data and validation")
    st.subheader("Every model uses the same data, the same split and the same metrics")
    st.markdown(
        f"In this tab we give the full results for readers who work in pricing. Every figure is on the holdout of "
        f"{holdout['policies']:,} policies ({holdout['claims']:,} claims) unless it says otherwise, and each model is "
        f"rebalanced by one scalar factor fitted on the training data, all within 0.3% of 1.")
    data_and_cleaning(data)
    split_and_validation(data)

    st.subheader("The holdout was used to score models, and once to choose between them")
    st.markdown(
        "We used the holdout in three stages. First, we scored seven models with all nine rating factors: the GLM, the "
        "GLM with an interaction between the no-claims score and region, the GLM with the ladder indicator, the GLM with "
        "both, the GBM, the additive GBM and the GBM with an increasing constraint on the score.")
    st.markdown(
        "Second, two things followed from those results. We dropped the interaction, since its interval included zero, "
        "and the ladder indicator's result prompted the investigation of the no-claims score, although the investigation "
        "itself used the training data only.")
    st.markdown(
        f"Third, we committed the specification of the two final models ([commit f954934]({SPEC_COMMIT_URL})) and scored "
        f"them once. No model was changed after that, and in particular we did not add the record length as a covariate, "
        f"although the calibration result suggests it. The final figures may therefore be mildly optimistic.")

    st.divider()
    section_label("Model performance")
    st.subheader("The final GLM and GBM are level on every holdout metric")
    st.markdown(
        "The table gives every model we scored on the holdout. The reduction in prediction error is against the GLM with "
        "the same rating factors, as a percentage of that GLM's error, and its interval is a paired bootstrap interval "
        "from 1,000 resamples of holdout policies.")
    rows = data["models"]["rows"]
    no_wrap = {"white-space": "nowrap"}  # only the model names wrap, never a number
    st.markdown("**Final models: eight rating factors, without the no-claims score**")
    st.table(models_table([r for r in rows if r["factor_set"] == "final"]).style.set_properties(**no_wrap))
    st.markdown("**First comparison: nine rating factors, with the no-claims score**")
    st.table(models_table([r for r in rows if r["factor_set"] == "with_score"]).style.set_properties(**no_wrap))
    st.caption(
        "Prediction error is the Poisson deviance per policy-year (lower is better), and A / E is actual claims over "
        "expected. The ladder indicator flags no-claims scores that can only be reached through a claim, and \"score by "
        "region\" is an interaction between the no-claims score and region. The additive GBM lets each tree use one "
        "factor only, so that it has no interactions, and the monotone GBM is constrained to predict more claims for a "
        "higher no-claims score. The nine-factor models are the comparison that exposed the timing problem, not pricing "
        "models.")

    st.subheader("The two final models rank drivers almost equally well")
    st.markdown(
        "We order the holdout policies from the lowest predicted claim rate to the highest and plot the cumulative share "
        "of exposure against the cumulative share of claims. A model with no ability to rank drivers follows the "
        "diagonal, and the Gini coefficient is twice the area between the curve and the diagonal.")
    show(lorenz_chart(data["lorenz"]))
    st.caption("Note that Gini measures ranking only. A model can rank well and still be mis-calibrated, which is why we "
               "report actual over expected alongside it.")

    st.divider()
    section_label("Diagnostics")
    st.subheader("Where the final models disagree, neither is consistently closer to the actual rate")
    st.markdown(
        "We sort the holdout policies by the ratio of the GBM's prediction to the GLM's and cut them into ten groups of "
        "equal exposure. Group 1 holds the policies where the GBM is furthest below the GLM, and group 10 those where it "
        "is furthest above.")
    show(double_lift_chart(data["double_lift"]["rows"]))
    st.caption(
        "For each group we plot the actual claim rate and each model's average prediction. Where the models disagree, "
        "the better model is the one whose line stays closer to the actual line. In both end groups, where they disagree "
        "most, the actual rate lies between the two models.")

    st.subheader("Both models over-predict full-year records and under-predict short ones")
    record = {r["record_length"]: r for r in data["record_length"]["rows"]}
    st.markdown(
        "Claim frequency is not proportional to the length of the record in this data, whereas both models assume that "
        "it is, since they take exposure as an offset with its coefficient fixed at 1. The standard remedy is to fit log "
        "exposure as a covariate, which we did not apply, since the final models had already been scored and the change "
        "would be one more choice made with the holdout in view.")
    show(record_length_chart(data["record_length"]["rows"]))
    st.caption(
        f"Holdout exposure: {record['under half a year']['exposure_share']:.0%} in records under half a year, "
        f"{record['half to one year']['exposure_share']:.0%} in records of half a year to a year and "
        f"{record['full year']['exposure_share']:.0%} in full-year records.")

    st.subheader("The final GLM can be read directly as a rating table")
    rel = data["relativities"]
    names = [f["name"] for f in rel["factors"]] + [rel["density"]["name"]]
    choice = st.selectbox("Rating factor", names, index=0)
    if choice == rel["density"]["name"]:
        show(density_chart(rel["density"]))
    else:
        show(relativity_chart(next(f for f in rel["factors"] if f["name"] == choice)))
    st.caption(
        "Each relativity multiplies the predicted claim rate against the reference level, which is the level with the "
        "most exposure. Population density enters on the log scale, and so its relativity is a smooth curve. Severity "
        "uses a separate Gamma GLM, shared by both models, with driver age as its only factor.")

    st.divider()
    section_label("The no-claims score")
    st.subheader("Three training-data checks point to probable leakage in the no-claims score")
    checks = data["timing_checks"]
    st.markdown(
        "The first check compares claim rates for no-claims scores below 100 on the claim-free path (the values a driver "
        "with no claims can hold) and off it (values that can only be reached through a claim), by length of record.")
    lengths = {"under half a year": "Under half a year", "half to one year": "Half a year to a year", "full year": "Full year"}
    check1 = {r["record_length"]: r for r in checks["check1_record_length"]}
    st.table(pd.DataFrame([{
        "Record length": lengths[k],
        "Claim rate on the path": f"{check1[k]['rate_on_path']:.3f}",
        "Claim rate off the path": f"{check1[k]['rate_off_path']:.3f}",
        "Ratio": f"{check1[k]['ratio']:.2f}",
        "GLM relativity": f"{check1[k]['glm_relativity']:.2f}",
    } for k in RECORD_LENGTHS]).set_index("Record length"))
    st.markdown(
        "The second gives, for full-year records, the share of drivers with a claim in the year. A driver at 50 who claims "
        "moves to 62, and then to 58, 55 and 52 over the following claim-free renewals.")
    labels = {"50": "50", "62": "62 (surcharged at the last renewal)", "58": "58 (one renewal later)",
              "55": "55 (two renewals later)", "52": "52 (three renewals later)",
              "one": "One surcharge above another claim-free value"}
    by_group = {r["group"].split(" ")[0]: r for r in checks["check2_full_year_share"]}
    rows = [by_group[k] for k in labels if k in by_group]
    st.table(pd.DataFrame([{
        "No-claims score": labels[r["group"].split(" ")[0]],
        "Full-year records": f"{r['policies']:,}",
        "Share with a claim": f"{r['share_with_claim']:.1%}",
    } for r in rows]).set_index("No-claims score"))
    st.markdown(
        "The third gives claims per driver per year in eight equal blocks of training policies, ordered by policy number, "
        "which appears to follow the order in which policies entered the file.")
    stair = pd.DataFrame(checks["check3_staircase"])
    stair["Block"] = stair["block"].str.split(" ").str[0]
    table = stair.pivot(index="Block", columns="value", values="rate")[["50", "62", "58", "55", "52"]]
    table.index = [f"{b} (oldest)" if b == "1" else f"{b} (newest)" if b == "8" else b for b in table.index]
    st.table(table.map(lambda v: f"{v:.3f}").rename_axis("Block"))
    later = stair[stair["value"].isin(["58", "55", "52"])]["claims"]
    st.caption(
        f"Older blocks show the excess at 62 and also at 58, 55 and 52, one more renewal back in each older block. The "
        f"cells for 58, 55 and 52 hold between {later.min()} and {later.max()} claims each, so single cells are "
        f"uncertain, but the pattern across the table is not.")

    st.divider()
    section_label("In practice")
    st.subheader("Insurers still price with GLMs because a GLM is a set of rating tables")
    st.markdown(
        "A GLM can be read factor by factor, and so an underwriter or a regulator can see why a price is what it is. In "
        "many markets regulators expect pricing factors to be explained and checked for unfair discrimination, which is "
        "far easier with a table than with many trees.")
    st.markdown(
        "Furthermore, rating engines are built to apply tables, and so a GLM can be implemented directly. Our Excel "
        "version of the final GLM reproduces the Python price for 100 of 100 holdout policies. In practice pricing teams "
        "often use a GBM to find structure that the GLM misses, such as a shape or an interaction, and then add it to the "
        "GLM, as we did with the ladder indicator.")


def main():
    st.set_page_config(page_title="GLM vs GBM | Leo Yu", layout="centered")
    data = load_data()
    missing = [name for name in REQUIRED_FILES if name not in data]
    if missing:
        st.error(f"Missing data files in artifacts/dashboard/: {', '.join(missing)}. "
                 f"Run `python src\\build_dashboard_data.py` from the repository root, then refresh the page.")
        st.stop()
    header(data)
    # Stage 2 adds a "Try it yourself" tab between these two.
    tab_article, tab_analysis = st.tabs(["Article", "Quantitative analysis"])
    with tab_article:
        article(data)
    with tab_analysis:
        quantitative_analysis(data)


if __name__ == "__main__":
    main()
