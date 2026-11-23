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


def test_interval_sentence():
    assert "excludes zero" in dashboard.interval_sentence({"deviance_reduction": 0.002, "bootstrap_95_low": 0.001, "bootstrap_95_high": 0.003})
    assert "not distinguishable from zero" in dashboard.interval_sentence({"deviance_reduction": 0.0003, "bootstrap_95_low": -0.0002, "bootstrap_95_high": 0.0008})
    assert "GLM scores better" in dashboard.interval_sentence({"deviance_reduction": -0.002, "bootstrap_95_low": -0.003, "bootstrap_95_high": -0.001})


def test_tables_shown_on_the_dashboard(toy_predictions):
    s = build_summary(toy_predictions)
    table = dashboard.metrics_table(s["final"]["models"])
    assert list(table.index) == ["GLM", "GBM"]
    assert table.loc["GBM", "Poisson deviance"] == s["final"]["models"]["gbm_final"]["poisson_deviance"]
    long = dashboard.double_lift_long(s["final"]["double_lift"])
    assert len(long) == 30 and set(long["series"]) == {"Actual", "GLM", "GBM"}


def test_dashboard_imports_no_modelling_library():
    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= {"json", "pathlib", "altair", "pandas", "streamlit"}, imported


def test_dashboard_runs_on_the_saved_results():
    if not dashboard.RESULTS_PATH.exists():
        pytest.skip("outputs/model_comparison.json not present")
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(DASHBOARD), default_timeout=60).run()
    assert not app.exception
    headers = [h.value for h in app.header]
    assert headers == ["Final models: eight rating factors, without BonusMalus", "With BonusMalus: the comparison that was set aside"]
    assert len(app.dataframe) == 2
