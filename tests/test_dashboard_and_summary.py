"""The results summary and the dashboard that reads it."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dashboard
import metrics
from summarise_models import build_summary

DASHBOARD = Path(dashboard.__file__)


@pytest.fixture(scope="module")
def toy_predictions():
    rng = np.random.default_rng(0)
    n = 20000
    exposure = rng.uniform(0.1, 1.0, n)
    true_rate = np.exp(-2.5 + 0.5 * rng.normal(size=n))
    df = pd.DataFrame({"is_holdout": rng.random(n) < 0.25, "Exposure": exposure, "ClaimNb": rng.poisson(true_rate * exposure)})
    df["TotalLoss"] = df["ClaimNb"] * rng.gamma(2.0, 1000.0, n)
    df["TotalLossCapped"] = df["TotalLoss"].clip(upper=5000.0)
    for name, noise in [("glm_final", 0.6), ("gbm_final", 0.4), ("glm", 0.3), ("gbm", 0.2)]:
        df[f"freq_{name}"] = true_rate * np.exp(noise * rng.normal(size=n))
        df[f"pp_{name}"] = df[f"freq_{name}"] * 1500.0
    return df


def test_summary_parts_and_numbers(toy_predictions):
    s = build_summary(toy_predictions)
    h = toy_predictions[toy_predictions["is_holdout"]]
    assert set(s["final"]["models"]) == {"glm_final", "gbm_final"}
    assert set(s["with_bonus_malus"]["models"]) == {"glm", "gbm"}
    assert s["other_models_scored_on_the_holdout"] == {}
    dev = {m: metrics.mean_poisson_deviance(h["ClaimNb"], h[f"freq_{m}"], h["Exposure"]) for m in ["glm_final", "gbm_final", "glm", "gbm"]}
    assert s["final"]["models"]["gbm_final"]["poisson_deviance"] == pytest.approx(dev["gbm_final"])
    assert s["final"]["glm_minus_gbm"]["deviance_reduction"] == pytest.approx(dev["glm_final"] - dev["gbm_final"])
    assert s["cost_of_leaving_out_bonus_malus"]["glm"]["deviance_reduction"] == pytest.approx(dev["glm_final"] - dev["glm"])
    assert s["with_bonus_malus"]["gap_closed"]["glm_to_gbm_gap_deviance"] == pytest.approx(dev["glm"] - dev["gbm"])
    lift = pd.DataFrame(s["final"]["double_lift"])
    assert list(lift["bin"]) == list(range(1, 11))
    assert {"actual_rate", "rate_glm", "rate_gbm"} <= set(lift.columns)
    assert s["holdout"]["policies"] == len(h)


def test_verdict():
    assert dashboard.verdict(1.08, 1.61) == "clearly and significantly better"
    assert dashboard.verdict(-0.04, 0.18) == "within the margin of error"
    assert dashboard.verdict(-0.3, -0.1) == "clearly and significantly worse"


def test_wording_helpers():
    rows = [{"key": "glm", "factor_set": "with_score", "label": "GLM", "reference": True, "poisson_deviance": 0.45,
             "deviance_explained": 0.05, "gini": 0.3, "a_over_e": 1.0},
            {"key": "gbm", "factor_set": "with_score", "label": "GBM", "reference": False, "poisson_deviance": 0.44,
             "deviance_explained": 0.06, "gini": 0.32, "a_over_e": 1.0, "reduction": 0.006, "pct": 1.3, "low_pct": 1.1, "high_pct": 1.6},
            {"key": "gbm_final", "factor_set": "final", "label": "GBM", "reference": False, "poisson_deviance": 0.46,
             "deviance_explained": 0.02, "gini": 0.18, "a_over_e": 1.0, "reduction": 0.0003, "pct": -0.07, "low_pct": -0.1, "high_pct": 0.2}]
    table = dashboard.models_table(rows)
    assert list(table["Against the GLM"]) == ["reference", "1.30% lower", "0.07% higher"]
    assert list(table["95% interval"]) == ["", "1.10% to 1.60%", "-0.10% to 0.20%"]
    assert dashboard.pct(-0.00195) == "-0.002%" and dashboard.pct(1.3454) == "1.35%" and dashboard.pct(0.0) == "0.00%"
    assert dashboard.times(3.03) == "three times" and dashboard.times(9.23) == "nine times" and dashboard.times(12.0) == "12 times"
    assert dashboard.share_of_lead_lost({"models": {"rows": rows}}) == pytest.approx(0.95)
    lift = [{"actual_rate": 0.1, "rate_glm": 0.2, "rate_gbm": 0.15}, {"actual_rate": 0.1, "rate_glm": 0.11, "rate_gbm": 0.2}]
    assert dashboard.gbm_closer_groups(lift) == 1
    stair = [{"block": "8 (IDpol from 1)", "value": v, "rate": r} for v, r in [("50", 0.05), ("62", 0.09), ("58", 0.1), ("55", 0.04), ("52", 0.05)]]
    stair += [{"block": "1 (IDpol from 2)", "value": "62", "rate": 0.5}]
    assert dashboard.staircase_newest_max_ratio({"timing_checks": {"check3_staircase": stair}}) == pytest.approx(2.0)


def test_load_data_rereads_added_or_changed_files(tmp_path):
    # The cache must not hide a file written after the app first loaded its data.
    (tmp_path / "a.json").write_text('{"x": 1}', encoding="utf-8")
    assert dashboard.load_data(tmp_path) == {"a": {"x": 1}}
    (tmp_path / "b.json").write_text('{"y": 2}', encoding="utf-8")
    assert set(dashboard.load_data(tmp_path)) == {"a", "b"}
    (tmp_path / "a.json").write_text('{"x": 10}', encoding="utf-8")
    assert dashboard.load_data(tmp_path)["a"] == {"x": 10}


def test_every_required_file_is_built():
    data = _dashboard_data()
    assert set(dashboard.REQUIRED_FILES) <= set(data)


def _dashboard_data():
    if not dashboard.DATA_DIR.exists():
        pytest.skip("artifacts/dashboard not present")
    return dashboard.load_data()


def test_headline_numbers_on_the_saved_files():
    data = _dashboard_data()
    gbm, gbm_final = dashboard.model(data, "gbm"), dashboard.model(data, "gbm_final")
    # The relative measure quoted on the page: about 1.3% with the no-claims score, 0.07% without.
    assert round(gbm["pct"], 1) == 1.3 and round(gbm_final["pct"], 2) == 0.07
    assert gbm["low_pct"] > 0 and gbm_final["low_pct"] < 0 < gbm_final["high_pct"]
    # "Over 90% of the GBM's lead in the first comparison goes once the score is removed."
    assert dashboard.share_of_lead_lost(data) > 0.9


def test_every_chart_builds_on_the_saved_files():
    data = _dashboard_data()
    charts = [dashboard.driver_age_chart(data["driver_age"]["rows"]), dashboard.shap_chart(data["shap"]["rows"]),
              dashboard.score_chart(data["score_rates"]["rows"]), dashboard.with_without_chart(data),
              dashboard.lift_chart(data["lift"]["rows"]),
              dashboard.price_ratio_chart(data["price_ratio"]), dashboard.double_lift_chart(data["double_lift"]["rows"]),
              dashboard.lorenz_chart(data["lorenz"]), dashboard.density_chart(data["relativities"]["density"]),
              dashboard.record_length_chart(data["record_length"]["rows"])]
    charts += [dashboard.relativity_chart(f) for f in data["relativities"]["factors"]]
    for chart in charts:
        spec = chart.to_dict()
        assert spec["title"]["text"]


def test_dashboard_imports_no_modelling_library():
    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= {"json", "pathlib", "textwrap", "altair", "pandas", "streamlit"}, imported


def test_dashboard_runs_on_the_saved_files():
    _dashboard_data()
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(DASHBOARD), default_timeout=120).run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Article", "Quantitative analysis"]
    assert [t.value for t in app.title] == ["GLM vs GBM"]
    labels = [m.value.split(">")[1].split("<")[0] for m in app.markdown if f"color:{dashboard.ACCENT}" in m.value]
    assert labels == ["Introduction", "Analysis", "Conclusion",
                      "Data and validation", "Model performance", "Diagnostics", "The no-claims score", "In practice"]
    subheaders = [h.value for h in app.subheader]
    assert subheaders[:6] == [
        "The two ways to predict car insurance claims",
        "At first, the GBM was clearly better, and it relied most on the no-claims score",
        "The no-claims score appears to contain information from after the claims",
        "Without it, the GBM's advantage all but disappears",
        "Without the no-claims score, both models agree on who is high risk",
        "Method and limitations",
    ]
    assert len(app.table) == 5
    conclusion = next(m.value for m in app.markdown if m.value.startswith("On this data, and without the no-claims score"))
    assert "within the margin of error" in conclusion
