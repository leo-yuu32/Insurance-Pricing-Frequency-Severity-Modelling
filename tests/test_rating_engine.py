"""The Excel rating engine for the final GLM."""

import json

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

import build_rating_engine as engine
from design import BANDS, _band_labels

FINAL_GLM = engine.OUTPUTS / "final_glm.json"


def test_lower_bound_lookup_reproduces_the_pipeline_bands():
    # Excel bands a value with LOOKUP on the lower bounds. For whole numbers this must give the
    # same band as pd.cut with right-closed edges, which the pipeline uses.
    for name in ["DrivAgeBand", "VehAgeBand", "VehPowerBand"]:
        source, edges = BANDS[name]
        values = pd.Series(np.arange(edges[0] + 1, edges[-1] + 1), dtype=float)
        expected = pd.cut(values, bins=edges, labels=_band_labels(edges)).astype(str).to_numpy()
        lower = engine.band_lower_bounds(edges)
        got = np.array(_band_labels(edges))[np.searchsorted(lower, values.to_numpy(), side="right") - 1]
        np.testing.assert_array_equal(got, expected, err_msg=name)


def test_factor_rows_include_reference_levels_with_zero():
    final_glm = {"frequency": {"coefficients": [{"term": "Intercept", "level": "", "coef": -2.0},
                                                {"term": "Colour", "level": "blue", "coef": 0.0},
                                                {"term": "Colour", "level": "red", "coef": 0.4},
                                                {"term": "LogDensity", "level": "", "coef": 0.1}]},
                 "severity": {"coefficients": [{"term": "Intercept", "level": "", "coef": 7.0}]}}
    keys = {key: coef for key, _, _, _, coef in engine.factor_rows(final_glm)}
    assert keys == {"frequency|Colour|blue": 0.0, "frequency|Colour|red": 0.4, "frequency|LogDensity": 0.1}
    assert engine.intercept(final_glm, "severity") == 7.0


@pytest.fixture(scope="module")
def final_glm():
    if not FINAL_GLM.exists():
        pytest.skip("outputs/final_glm.json not present")
    return json.loads(FINAL_GLM.read_text(encoding="utf-8"))


def test_rating_tables_reproduce_the_pipeline_prices(final_glm):
    if not engine.PREDICTIONS_PATH.exists():
        pytest.skip("outputs/predictions.parquet not present")
    p = pd.read_parquet(engine.PREDICTIONS_PATH)
    h = p[p["is_holdout"]]
    np.testing.assert_allclose(engine.python_price(final_glm, h), h["pp_glm_final"] * final_glm["large_loss_load"], rtol=1e-12)


def test_excel_reproduces_python_for_every_reconcile_policy():
    if not engine.OUTPUT_PATH.exists():
        pytest.skip("workbook not built")
    ws = load_workbook(engine.OUTPUT_PATH, data_only=True)["Reconcile"]
    names = [c.value for c in ws[1]]
    rows = list(ws.iter_rows(min_row=2, max_row=engine.N_POLICIES + 1, values_only=True))
    if rows[0][names.index("Match")] is None:
        pytest.skip("workbook has no values computed by Excel (build it with: python src/build_rating_engine.py recalc)")
    excel = np.array([r[names.index("Pure premium")] for r in rows], dtype=float)
    python = np.array([r[names.index("Python pure premium")] for r in rows], dtype=float)
    assert all(r[names.index("Match")] is True for r in rows)
    np.testing.assert_allclose(excel, python, rtol=engine.MATCH_TOLERANCE)


def test_legacy_workbook_is_left_alone():
    assert engine.OUTPUT_PATH.name != "rating_engine.xlsx"
