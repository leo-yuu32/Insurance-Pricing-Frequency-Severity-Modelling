"""The results summary and the dashboard that reads it."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dashboard
import metrics
from run_glm import PREDICTIONS_PATH
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
    assert dashboard.lift_spreads(data["lift"]["rows"])[0] == "three times"
    assert dashboard.times(dashboard.shap_ratio(data["shap"]["rows"])) == "three times"
    for chart in charts:
        spec = chart.to_dict()
        assert spec["title"]["text"]


def test_dashboard_trains_nothing():
    # lightgbm is imported only to load the saved final GBM for prediction; nothing is fitted in the app.
    source = DASHBOARD.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= {"json", "pathlib", "textwrap", "altair", "lightgbm", "numpy", "pandas", "streamlit"}, imported
    assert ".train(" not in source and ".fit(" not in source and "Dataset(" not in source


def test_bands_and_area():
    band = {"edges": [17, 19, 21, 200], "labels": ["18-19", "20-21", "22+"]}
    assert [dashboard.band_of(v, band) for v in (18, 19, 20, 21, 22, 90)] == ["18-19", "18-19", "20-21", "20-21", "22+", "22+"]
    bounds = [{"area": "A", "min": 1, "max": 50}, {"area": "B", "min": 50, "max": 100}, {"area": "D", "min": 500, "max": 1993},
              {"area": "E", "min": 2001, "max": 9850}]
    assert [dashboard.area_from_density(d, bounds) for d in (8, 60, 1600, 1997, 3200)] == ["A", "B", "D", "D", "E"]


def test_largest_model_gap():
    rows = [{"level": "a", "exposure": 5000, "glm": 0.10, "gbm": 0.104}, {"level": "b", "exposure": 3000, "glm": 0.10, "gbm": 0.095},
            {"level": "c", "exposure": 100, "glm": 0.10, "gbm": 0.2}]
    gap, level = dashboard.largest_model_gap(rows)
    assert gap == pytest.approx(0.05) and level == "b"
    assert dashboard.largest_model_gap(rows[2:]) is None


def _pricing():
    data = _dashboard_data()
    if not dashboard.GBM_MODEL_PATH.exists():
        pytest.skip("artifacts/gbm_final.txt not present")
    booster, info = dashboard.load_gbm()
    return data["rating_table"], booster, info


def test_waterfall_adds_up_to_the_glm_price():
    table, _, _ = _pricing()
    inputs = table["inputs"]
    driver = {"DrivAge": 19, "VehAge": 2, "VehPower": 11, "Density": 6400, "VehBrand": "B12", "VehGas": "Diesel",
              "Region": "R11", "Area": "E"}
    r = dashboard.glm_price(table, driver)
    assert r["steps"][0]["start"] == r["base"] and r["steps"][-1]["end"] == pytest.approx(r["price"])
    assert all(a["end"] == b["start"] for a, b in zip(r["steps"], r["steps"][1:]))
    scale = table["glm_pure_premium_rebalance_factor"] * table["large_loss_load"]
    assert r["price"] == pytest.approx(r["frequency"] * r["severity"] * scale)
    typical = {"DrivAge": inputs["driver_age"]["default"], "VehAge": inputs["vehicle_age"]["default"],
               "VehPower": inputs["vehicle_power"]["default"], "Density": inputs["density"]["default"],
               "VehBrand": inputs["brand_default"], "VehGas": inputs["fuel_default"], "Region": inputs["region_default"]}
    assert len(dashboard.waterfall_chart(dashboard.glm_price(table, typical)).to_dict()["layer"]) == 2
    # Driver age is split into the claim-rate step and the cost-per-claim step.
    assert [s["step"] for s in r["steps"][:2]] == ["Driver age: claim rate", "Driver age: cost per claim"]
    assert r["steps"][0]["factor"] * r["steps"][1]["factor"] == pytest.approx(
        table["frequency"]["relativities"]["DrivAgeBand"]["18-19"] * table["severity"]["relativities"]["DrivAgeBand"]["18-19"])


def test_waterfall_euros_add_up_to_the_displayed_price():
    table, _, _ = _pricing()
    inputs = table["inputs"]
    rng = np.random.default_rng(0)
    for _ in range(200):
        driver = {"DrivAge": int(rng.integers(18, 91)), "VehAge": int(rng.integers(0, 26)), "VehPower": int(rng.integers(4, 16)),
                  "Density": int(rng.choice(inputs["density"]["options"])),
                  "VehBrand": rng.choice([b["code"] for b in inputs["brands"]]),
                  "VehGas": rng.choice(["Regular", "Diesel"]), "Region": rng.choice([x["code"] for x in inputs["regions"]])}
        d = dashboard.waterfall_rows(dashboard.glm_price(table, driver))
        base, steps, final = d["euros"].iloc[0], d["euros"].iloc[1:-1], d["euros"].iloc[-1]
        assert base + steps.sum() == final
        assert d["label"].iloc[-1] == f"EUR {final:,}"
    labels = dashboard.waterfall_rows({"base": 100.0, "price": 140.0, "steps": [
        {"step": "Region", "term": "Region", "level": "R11", "factor": 1.4, "start": 100.0, "end": 140.0}]})["label"]
    assert list(labels) == ["EUR 100", "×1.40 (+40)", "EUR 140"]
    assert dashboard.round_ticks(867) == [0, 200, 400, 600, 800] and dashboard.round_ticks(144) == [0, 50, 100]


def test_app_prices_match_the_pipeline():
    # The relativity table and the saved GBM must give exactly the prices the pipeline computed.
    table, booster, info = _pricing()
    path = PREDICTIONS_PATH
    if not path.exists():
        pytest.skip("outputs/predictions.parquet not present")
    p = pd.read_parquet(path)
    h = p[p["is_holdout"]].sample(300, random_state=0)
    drivers = [{"DrivAge": int(r.DrivAge), "VehAge": int(r.VehAge), "VehPower": int(r.VehPower), "Density": float(r.Density),
                "VehBrand": r.VehBrand, "VehGas": r.VehGas, "Region": r.Region, "Area": r.Area} for r in h.itertuples()]
    load = table["large_loss_load"]
    glm = np.array([dashboard.glm_price(table, d)["price"] for d in drivers])
    gbm = np.array([dashboard.gbm_price(booster, info, table, d)["price"] for d in drivers])
    assert np.allclose(glm, h["pp_glm_final"] * load, rtol=1e-9, atol=0)
    assert np.allclose(gbm, h["pp_gbm_final"] * load, rtol=1e-9, atol=0)
    # In the data the area type is a banding of density; the app derives it from the density input.
    pairs = p[~p["Density"].isin([50, 100, 500])].drop_duplicates(["Density", "Area"])
    derived = [dashboard.area_from_density(d, table["inputs"]["area_bounds"]) for d in pairs["Density"]]
    assert derived == list(pairs["Area"])


def test_dashboard_runs_on_the_saved_files():
    _dashboard_data()
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(DASHBOARD), default_timeout=120).run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Article", "Try it yourself", "Quantitative analysis"]
    assert [t.value for t in app.title] == ["GLM vs GBM"]
    labels = [m.value.split(">")[1].split("<")[0] for m in app.markdown if f"color:{dashboard.ACCENT}" in m.value]
    assert labels == ["Introduction", "Analysis", "Conclusion", "Price a driver", "Explore the data",
                      "Data and validation", "Model performance", "Diagnostics", "The no-claims score", "In practice"]
    assert [m.label for m in app.metric] == ["GLM", "GBM"]
    before = app.metric[0].value
    app.selectbox(key="region").set_value("R11").run()
    app.selectbox(key="explore_factor").set_value("Region").run()
    assert not app.exception and app.metric[0].value != before
    subheaders = [h.value for h in app.subheader]
    assert subheaders == [
        "The two ways to predict car insurance claims", "The first comparison", "A closer look at the no-claims score",
        "The final comparison", "How the models rank drivers", "Limitations", "What this means",
        "Where the price comes from", "Predictions by rating factor",
        "The data", "The train and holdout split", "How the holdout was used", "Prediction error", "Lorenz curves and Gini",
        "Double-lift chart", "Actual vs expected by record length", "GLM relativities", "Three timing checks",
        "Why insurers still use GLMs",
    ]
    assert len(app.table) == 5
    conclusion = next(m.value for m in app.markdown if m.value.startswith("On this data, and without the no-claims score"))
    assert "within the margin of error" in conclusion
