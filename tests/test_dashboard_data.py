"""The functions that precompute the dashboard's numbers (src/build_dashboard_data.py)."""

import json
import math

import numpy as np
import pandas as pd
import pytest

import build_dashboard_data as bd
import metrics
from run_glm import OUTPUTS


def test_relative_reduction_is_a_share_of_the_reference_deviance():
    r = bd.relative_reduction({"deviance_reduction": 0.006, "bootstrap_95_low": 0.004, "bootstrap_95_high": 0.008}, 0.4)
    assert r == pytest.approx({"pct": 1.5, "low_pct": 1.0, "high_pct": 2.0})


def test_interval_accepts_both_saved_layouts():
    a = bd._interval({"estimate": 1.0, "bootstrap_95_low": 0.5, "bootstrap_95_high": 1.5})
    b = bd._interval({"deviance_reduction": 1.0, "bootstrap_95_low": 0.5, "bootstrap_95_high": 1.5, "glm_deviance": 9.0})
    assert a == b == {"deviance_reduction": 1.0, "bootstrap_95_low": 0.5, "bootstrap_95_high": 1.5}


def test_rates_by_weights_by_exposure_and_keeps_the_range():
    df = pd.DataFrame({"age": [18, 18, 19, 90], "Exposure": [1.0, 0.5, 1.0, 1.0], "ClaimNb": [1, 0, 0, 3],
                       "p": [0.2, 0.4, 0.1, 0.5]})
    out = bd.rates_by(df, "age", {"model": "p"}, 18, 85).set_index("age")
    assert list(out.index) == [18, 19]
    assert out.loc[18, "actual"] == pytest.approx(1 / 1.5)
    assert out.loc[18, "model"] == pytest.approx((0.2 * 1.0 + 0.4 * 0.5) / 1.5)


def test_claim_rate_by_score_uses_full_year_records_only():
    df = pd.DataFrame({"BonusMalus": [50, 50, 50, 62, 62, 62, 70],
                       "Exposure": [1.0, 1.0, 0.4, 1.0, 1.0, 1.0, 1.0],
                       "ClaimNb": [0, 1, 5, 1, 1, 0, 0]})
    out = bd.claim_rate_by_score(df, 50, 65, min_policies=2).set_index("score")
    assert list(out.index) == [50, 62]
    assert out.loc[50, "policies"] == 2 and out.loc[50, "rate"] == pytest.approx(0.5)
    assert out.loc[62, "rate"] == pytest.approx(2 / 3)


def test_price_ratio_summary():
    ratio = np.array([0.5, 0.85, 0.95, 1.0, 1.05, 1.15, 2.0])
    exposure = np.array([1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0])
    edges = np.round(np.arange(0.60, 1.5001, 0.02), 2)
    s = bd.price_ratio_summary(ratio, exposure, edges, 0.10)
    assert sum(b["share"] for b in s["bins"]) == pytest.approx(1.0)
    assert s["share_higher"] == pytest.approx(2 / 8) and s["share_lower"] == pytest.approx(2 / 8)
    assert s["share_differing"] == pytest.approx(4 / 8)
    assert s["share_beyond_axis"] == pytest.approx(2 / 8)
    flagged = sum(b["share"] for b in s["bins"] if b["differs"])
    assert flagged == pytest.approx(s["share_differing"])


def test_lift_by_average():
    # Four policies of equal exposure: the average prediction orders them 2, 0, 3, 1.
    a = np.array([0.2, 0.9, 0.1, 0.5])
    b = np.array([0.4, 0.7, 0.1, 0.3])
    out = bd.lift_by_average(a, b, actual=[1, 2, 0, 1], exposure=[1.0, 1.0, 1.0, 1.0], n_bins=2)
    assert list(out["bin"]) == [1, 2] and list(out["exposure"]) == [2.0, 2.0]
    assert list(out["actual_rate"]) == pytest.approx([0.5, 1.5])
    assert list(out["rate_a"]) == pytest.approx([0.15, 0.7]) and list(out["rate_b"]) == pytest.approx([0.25, 0.5])
    rng = np.random.default_rng(2)
    rate, exposure = np.exp(rng.normal(-2.5, 0.5, 4000)), rng.uniform(0.1, 1.0, 4000)
    out = bd.lift_by_average(rate, rate * 1.1, rng.poisson(rate * exposure), exposure, n_bins=10)
    assert len(out) == 10 and out["rate_a"].is_monotonic_increasing
    assert out["exposure"].min() > 0.95 * out["exposure"].mean()


def test_lorenz_points_match_the_gini():
    rng = np.random.default_rng(1)
    exposure = rng.uniform(0.1, 1.0, 5000)
    rate = np.exp(rng.normal(-2.5, 0.5, 5000))
    claims = rng.poisson(rate * exposure)
    x, y = bd.lorenz_points(rate, claims, exposure, 2001)
    assert (x[0], y[0], x[-1], y[-1]) == (0.0, 0.0, 1.0, 1.0)
    area = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
    assert 1 - 2 * area == pytest.approx(metrics.gini(rate, claims, exposure), abs=1e-3)
    x, y = bd.lorenz_points(np.ones(5000), claims, exposure, 11)
    assert np.allclose(x, y)


def test_glm_relativities():
    coefficients = [
        {"term": "Intercept", "level": "", "coef": -3.0},
        {"term": "DrivAgeBand", "level": "41-50", "coef": 0.0},
        {"term": "DrivAgeBand", "level": "18-19", "coef": 1.0},
        {"term": "DrivAgeBand", "level": "76+", "coef": -0.1},
        {"term": "LogDensity", "level": "", "coef": 0.1},
        {"term": "Region", "level": "R24", "coef": 0.0},
        {"term": "Region", "level": "R11", "coef": 0.3},
        {"term": "Region", "level": "R41", "coef": -0.2},
    ]
    out = bd.glm_relativities(coefficients, [10.0, 100.0, 1000.0], 100.0)
    age = next(f for f in out["factors"] if f["term"] == "DrivAgeBand")
    assert [r["level"] for r in age["levels"]] == ["18-19", "41-50", "76+"]
    assert age["levels"][0]["relativity"] == pytest.approx(math.e)
    assert [r["reference"] for r in age["levels"]] == [False, True, False]
    region = next(f for f in out["factors"] if f["term"] == "Region")
    assert [r["level"] for r in region["levels"]] == ["R11", "R24", "R41"]
    assert out["density"]["per_tenfold"] == pytest.approx(10 ** 0.1)
    assert [c["relativity"] for c in out["density"]["curve"]] == pytest.approx([10 ** -0.1, 1.0, 10 ** 0.1])


def test_rating_table_is_the_glm_as_relativities():
    final_glm = {
        "frequency": {"coefficients": [{"term": "Intercept", "level": "", "coef": -3.0},
                                       {"term": "DrivAgeBand", "level": "41-50", "coef": 0.0},
                                       {"term": "DrivAgeBand", "level": "18-19", "coef": 1.0},
                                       {"term": "LogDensity", "level": "", "coef": 0.1}]},
        "severity": {"coefficients": [{"term": "Intercept", "level": "", "coef": 7.0},
                                      {"term": "DrivAgeBand", "level": "41-50", "coef": 0.0}]},
        "bands": {"DrivAgeBand": [17, 19, 21, 23, 25, 30, 40, 50, 60, 75, 200]},
        "band_sources": {"DrivAgeBand": "DrivAge"},
        "pure_premium_rebalance_factor": 1.002, "large_loss_load": 1.19,
    }
    t = bd.rating_table(final_glm, 1.003)
    assert t["frequency"]["base"] == pytest.approx(math.exp(-3.0)) and t["severity"]["base"] == pytest.approx(math.exp(7.0))
    assert t["frequency"]["relativities"]["DrivAgeBand"] == pytest.approx({"41-50": 1.0, "18-19": math.e})
    assert t["frequency"]["log_density_coefficient"] == 0.1
    assert t["bands"]["DrivAgeBand"]["labels"][0] == "18-19" and t["bands"]["DrivAgeBand"]["labels"][-1] == "76+"
    assert (t["glm_pure_premium_rebalance_factor"], t["gbm_pure_premium_rebalance_factor"], t["large_loss_load"]) == (1.002, 1.003, 1.19)


def test_pricing_inputs_are_a_typical_driver():
    train = pd.DataFrame({"DrivAge": [30, 44, 60], "VehAge": [1, 6, 9], "VehPower": [5, 6, 7], "Density": [20, 393, 3000],
                          "VehBrand": ["B12", "B2", "B2"], "VehGas": ["Diesel", "Regular", "Regular"],
                          "Region": ["R24", "R11", "R24"], "Area": ["A", "C", "E"], "Exposure": [0.5, 1.0, 1.0]})
    i = bd.pricing_inputs(train)
    assert (i["driver_age"]["default"], i["vehicle_age"]["default"], i["vehicle_power"]["default"]) == (44, 6, 6)
    assert i["density"]["default"] == 393 and 393 in i["density"]["options"]
    assert not {50, 100, 500} & set(i["density"]["options"])
    assert [b["label"] for b in i["brands"]] == ["Brand group 2", "Brand group 12"]
    assert (i["brand_default"], i["fuel_default"], i["region_default"]) == ("B2", "Regular", "R24")
    assert [r["label"] for r in i["regions"]] == ["Centre (R24)", "Île-de-France (R11)"]
    assert sorted(["Rhône-Alpes", "Île-de-France", "Alsace"], key=bd._alphabetical) == ["Alsace", "Île-de-France", "Rhône-Alpes"]


def test_explore_levels_and_rates():
    df = pd.DataFrame({"DrivAge": [18, 20, 21, 90], "Density": [5, 10, 400, 20000], "Exposure": [1.0, 1.0, 0.5, 1.0],
                       "ClaimNb": [1, 0, 1, 0], "p": [0.2, 0.1, 0.3, 0.05], "VehBrand": ["B10", "B2", "B2", "B1"]})
    levels, order = bd.explore_levels(df, "DrivAge")
    assert list(levels) == ["18-20", "18-20", "21-25", "81+"] and order[0] == "18-20"
    levels, order = bd.explore_levels(df, "Density")
    assert list(levels) == ["Under 10", "10 to 30", "300 to 1,000", "10,000 and over"]
    levels, order = bd.explore_levels(df, "VehBrand")
    assert order == ["Group 1", "Group 2", "Group 10"]
    levels, order = bd.explore_levels(df, "DrivAge")
    rows = {r["level"]: r for r in bd.rates_by_level(df, levels, order, {"model": "p"})}
    assert rows["18-20"]["actual"] == pytest.approx(0.5) and rows["18-20"]["model"] == pytest.approx(0.15)
    assert list(rows) == ["18-20", "21-25", "81+"]


def _saved(name):
    path = OUTPUTS / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    return json.loads(path.read_text(encoding="utf-8"))


def test_model_rows_copy_the_saved_results():
    names = ["model_comparison", "metrics_glm", "metrics_gbm", "metrics_interactions", "metrics_final"]
    saved = {n: _saved(n) for n in names}
    rows = {r["key"]: r for r in bd.model_rows(saved)}
    comparison = saved["model_comparison"]
    for key in ["glm_final", "gbm_final"]:
        assert rows[key]["poisson_deviance"] == comparison["final"]["models"][key]["poisson_deviance"]
    for key in ["glm", "glm_ladder", "gbm_additive", "gbm"]:
        assert rows[key]["poisson_deviance"] == comparison["with_bonus_malus"]["models"][key]["poisson_deviance"]
    assert rows["glm_final"]["reference"] and rows["glm"]["reference"]
    assert rows["gbm_final"]["reduction"] == comparison["final"]["glm_minus_gbm"]["deviance_reduction"]
    assert rows["gbm"]["reduction"] == comparison["with_bonus_malus"]["differences"]["glm_to_gbm"]["deviance_reduction"]
    assert rows["gbm"]["pct"] == pytest.approx(100 * rows["gbm"]["reduction"] / rows["glm"]["poisson_deviance"])


def test_timing_checks_from_the_saved_results():
    out = bd.timing_checks(_saved("bonus_malus_timing"))
    for r in out["check1_record_length"]:
        assert r["ratio"] == pytest.approx(r["rate_off_path"] / r["rate_on_path"])
    assert {r["value"] for r in out["check3_staircase"]} == {"50", "62", "58", "55", "52"}
    assert len(out["check3_staircase"]) == 8 * 5
