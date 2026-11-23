"""Build the Excel rating engine for the final GLM.

Run from the repo root, after run_final.py:
    python src/build_rating_engine.py             build the workbook
    python src/build_rating_engine.py recalc      build it, then open it in Excel (Windows), recalculate and save,
                                                  so the file also stores the values Excel computed

Reads  outputs/final_glm.json and outputs/predictions.parquet.
Writes artifacts/rating_engine_final_glm.xlsx. This is a new file: the legacy workbook
artifacts/rating_engine.xlsx, with its hand-built Sample_Policy and Sensitivity_OneWay sheets, is
never opened or changed.

Every price in the workbook is an Excel formula on the rating tables, so the workbook is an
independent check of the Python pipeline: the Reconcile sheet compares Excel's price with
Python's for 100 holdout policies.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.workbook.defined_name import DefinedName

from config import SEED
from design import _band_labels
from run_glm import OUTPUTS, PREDICTIONS_PATH

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "artifacts" / "rating_engine_final_glm.xlsx"
N_POLICIES = 100
MATCH_TOLERANCE = 1e-9  # relative
# The raw inputs, in the order they appear on the Calculator and Reconcile sheets.
INPUTS = ["DrivAge", "VehAge", "VehPower", "Density", "VehBrand", "VehGas", "Region"]

BOLD = Font(bold=True)
HEADER = Font(bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", start_color="305496")
INPUT_FONT = Font(color="0000FF")              # blue: a value you can change
INPUT_FILL = PatternFill("solid", start_color="FFF2CC")


# ---------------------------------------------------------------------------------------------
# The rating tables, and the same calculation in Python (used by the tests)
# ---------------------------------------------------------------------------------------------

def band_lower_bounds(edges):
    """Lower bound of each band for whole-number inputs. A band (lo, hi] holds the integers lo+1 to hi."""
    return [int(e) + 1 for e in edges[:-1]]


def factor_rows(final_glm):
    """One row per model, term and level: (key, model, term, level, coefficient). Reference levels have 0."""
    rows = []
    for model in ["frequency", "severity"]:
        for c in final_glm[model]["coefficients"]:
            if c["term"] == "Intercept":
                continue
            key = f"{model}|{c['term']}|{c['level']}" if c["level"] else f"{model}|{c['term']}"
            rows.append((key, model, c["term"], c["level"], c["coef"]))
    return rows


def intercept(final_glm, model):
    return next(c["coef"] for c in final_glm[model]["coefficients"] if c["term"] == "Intercept")


def python_price(final_glm, policies):
    """The workbook's calculation done in Python from the same tables: pure premium per policy-year,
    including the large-loss load."""
    coef = {key: value for key, _, _, _, value in factor_rows(final_glm)}
    bands = {}
    for name, edges in final_glm["bands"].items():
        lower = band_lower_bounds(edges)
        idx = np.searchsorted(lower, policies[final_glm["band_sources"][name]].to_numpy(), side="right") - 1
        bands[name] = np.array(_band_labels(edges))[idx]
    rates = {}
    for model in ["frequency", "severity"]:
        eta = np.full(len(policies), intercept(final_glm, model))
        for c in final_glm[model]["coefficients"]:
            if c["term"] == "LogDensity":
                eta += c["coef"] * np.log(policies["Density"].to_numpy())
        for term in dict.fromkeys(c["term"] for c in final_glm[model]["coefficients"] if c["level"]):
            values = bands[term] if term in bands else policies[term].astype(str).to_numpy()
            eta += np.array([coef[f"{model}|{term}|{v}"] for v in values])
        rates[model] = np.exp(eta)
    return rates["frequency"] * rates["severity"] * final_glm["pure_premium_rebalance_factor"] * final_glm["large_loss_load"]


# ---------------------------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------------------------

def header(ws, row, names):
    for col, name in enumerate(names, start=1):
        cell = ws.cell(row=row, column=col, value=name)
        cell.font, cell.fill = HEADER, HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)


def name_range(wb, name, ref):
    wb.defined_names[name] = DefinedName(name=name, attr_text=ref)


def build_readme(wb, final_glm):
    ws = wb.create_sheet("README")
    lines = [
        ("Rating engine: final GLM", None),
        ("", None),
        ("What it does", "Prices a motor third-party liability policy from its rating factors with the final frequency GLM "
                         "(Poisson) and severity GLM (Gamma), both fitted with statsmodels. BonusMalus is not used (see the Caveats section of the project README)."),
        ("Price", "Pure premium per policy-year = frequency x severity x rebalancing factor x large-loss load."),
        ("Factors", "The rating tables: one row per term and level, with its coefficient and relativity. Reference levels have relativity 1."),
        ("Bands", "How driver age, vehicle age and vehicle power are banded."),
        ("Scalars", "Intercepts, the pure premium rebalancing factor and the large-loss load."),
        ("Calculator", "One policy: change the blue cells and read the price."),
        ("Reconcile", f"{N_POLICIES} holdout policies: the Excel price against the Python price. Match is TRUE when they agree to {MATCH_TOLERANCE:g} relative."),
        ("Colour key", "Blue on yellow = input you can change. Black = formula."),
        ("Regenerate", "python src/build_rating_engine.py recalc   (from the repo root, after the model scripts)"),
        ("Legacy", "artifacts/rating_engine.xlsx is the workbook for the earlier, legacy model, with the hand-built Sample_Policy and Sensitivity_OneWay sheets. It is kept unchanged."),
    ]
    for r, (a, b) in enumerate(lines, start=1):
        ws.cell(row=r, column=1, value=a).font = BOLD
        if b:
            ws.cell(row=r, column=2, value=b).alignment = Alignment(wrap_text=True, vertical="top")
    ws["A1"].font = Font(bold=True, size=14)
    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 110


def build_factors(wb, final_glm):
    ws = wb.create_sheet("Factors")
    header(ws, 1, ["Key", "Model", "Term", "Level", "Coefficient", "Relativity"])
    rows = factor_rows(final_glm)
    for r, (key, model, term, level, coef) in enumerate(rows, start=2):
        ws.append([key, model, term, level, coef, f"=EXP(E{r})"])
    last = len(rows) + 1
    name_range(wb, "FactorKeys", f"Factors!$A$2:$A${last}")
    name_range(wb, "FactorCoef", f"Factors!$E$2:$E${last}")
    for col, width in zip("ABCDEF", [36, 11, 14, 10, 14, 11]):
        ws.column_dimensions[col].width = width
    ws.freeze_panes = "A2"


def build_bands(wb, final_glm):
    ws = wb.create_sheet("Bands")
    col = 1
    for name, edges in final_glm["bands"].items():
        source = final_glm["band_sources"][name]
        ws.cell(row=1, column=col, value=f"{name} (from {source})").font = BOLD
        ws.cell(row=2, column=col, value="Lower bound").font = BOLD
        ws.cell(row=2, column=col + 1, value="Band").font = BOLD
        lower, labels = band_lower_bounds(edges), _band_labels(edges)
        for i, (lo, label) in enumerate(zip(lower, labels), start=3):
            ws.cell(row=i, column=col, value=lo)
            ws.cell(row=i, column=col + 1, value=str(label))
        a, b = ws.cell(row=3, column=col).column_letter, ws.cell(row=3, column=col + 1).column_letter
        end = 2 + len(lower)
        name_range(wb, f"{name}_Lower", f"Bands!${a}$3:${a}${end}")
        name_range(wb, f"{name}_Label", f"Bands!${b}$3:${b}${end}")
        ws.column_dimensions[a].width = 24
        ws.column_dimensions[b].width = 10
        col += 3


def build_scalars(wb, final_glm):
    ws = wb.create_sheet("Scalars")
    header(ws, 1, ["Name", "Value", "Meaning"])
    values = [
        ("FreqIntercept", intercept(final_glm, "frequency"), "Frequency GLM intercept (log claims per policy-year at the reference levels, before LogDensity)"),
        ("SevIntercept", intercept(final_glm, "severity"), "Severity GLM intercept (log euros per claim at the reference driver age band)"),
        ("PPFactor", final_glm["pure_premium_rebalance_factor"], "Rebalances frequency x severity to the capped training losses"),
        ("LargeLossLoad", final_glm["large_loss_load"], "Training losses over capped training losses: the cost of claims above the cap, spread evenly"),
    ]
    for r, (name, value, meaning) in enumerate(values, start=2):
        ws.cell(row=r, column=1, value=name)
        ws.cell(row=r, column=2, value=float(value))
        ws.cell(row=r, column=3, value=meaning)
        name_range(wb, name, f"Scalars!$B${r}")
    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 95


def frequency_formula(cells, final_glm):
    """Excel linear predictor of the frequency GLM. `cells` maps an input or band name to its cell."""
    parts = ["FreqIntercept"]
    for c in final_glm["frequency"]["coefficients"]:
        if c["term"] == "LogDensity":
            parts.append(f'INDEX(FactorCoef,MATCH("frequency|LogDensity",FactorKeys,0))*LN({cells["Density"]})')
    for term in dict.fromkeys(c["term"] for c in final_glm["frequency"]["coefficients"] if c["level"]):
        parts.append(f'INDEX(FactorCoef,MATCH("frequency|{term}|"&{cells[term]},FactorKeys,0))')
    return "=" + "+".join(parts)


def severity_formula(cells, final_glm):
    parts = ["SevIntercept"]
    for term in dict.fromkeys(c["term"] for c in final_glm["severity"]["coefficients"] if c["level"]):
        parts.append(f'INDEX(FactorCoef,MATCH("severity|{term}|"&{cells[term]},FactorKeys,0))')
    return "=" + "+".join(parts)


def band_formula(final_glm, name, value_cell):
    return f"=LOOKUP({value_cell},{name}_Lower,{name}_Label)"


def build_calculator(wb, final_glm, example):
    ws = wb.create_sheet("Calculator")
    ws["A1"] = "One policy"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A3"], ws["B3"] = "Input", "Value"
    ws["A3"].font = ws["B3"].font = BOLD
    cells = {}
    for r, name in enumerate(INPUTS, start=4):
        ws.cell(row=r, column=1, value=name)
        value = example[name]
        cell = ws.cell(row=r, column=2, value=float(value) if name in ("DrivAge", "VehAge", "VehPower", "Density") else str(value))
        cell.font, cell.fill = INPUT_FONT, INPUT_FILL
        cells[name] = f"B{r}"
    r = 4 + len(INPUTS) + 1
    ws.cell(row=r, column=1, value="Band").font = BOLD
    for name in final_glm["bands"]:
        r += 1
        ws.cell(row=r, column=1, value=name)
        ws.cell(row=r, column=2, value=band_formula(final_glm, name, cells[final_glm["band_sources"][name]]))
        cells[name] = f"B{r}"
    r += 2
    ws.cell(row=r, column=1, value="Output").font = BOLD
    f, s, pp, total = (f"B{r + i}" for i in range(1, 5))
    outputs = [
        ("Frequency (claims per policy-year)", "=EXP(" + frequency_formula(cells, final_glm)[1:] + ")", "0.00000"),
        ("Severity (euros per claim, capped basis)", "=EXP(" + severity_formula(cells, final_glm)[1:] + ")", "#,##0.00"),
        ("Pure premium, capped basis", f"={f}*{s}*PPFactor", "#,##0.00"),
        ("Pure premium including large-loss load", f"={pp}*LargeLossLoad", "#,##0.00"),
    ]
    for (label, formula, fmt), ref in zip(outputs, [f, s, pp, total]):
        ws[f"A{ref[1:]}"] = label
        ws[ref] = formula
        ws[ref].number_format = fmt
    ws.column_dimensions["A"].width = 42
    ws.column_dimensions["B"].width = 18


def build_reconcile(wb, final_glm, sample):
    ws = wb.create_sheet("Reconcile")
    bands = list(final_glm["bands"])
    names = ["IDpol"] + INPUTS + bands + ["Frequency", "Severity", "Pure premium", "Python pure premium", "Match"]
    header(ws, 1, names)
    col = {name: ws.cell(row=1, column=i).column_letter for i, name in enumerate(names, start=1)}
    for r, (_, p) in enumerate(sample.iterrows(), start=2):
        ws[f"{col['IDpol']}{r}"] = int(p["IDpol"])
        for name in INPUTS:
            ws[f"{col[name]}{r}"] = float(p[name]) if name in ("DrivAge", "VehAge", "VehPower", "Density") else str(p[name])
        cells = {name: f"{col[name]}{r}" for name in INPUTS}
        for name in bands:
            ws[f"{col[name]}{r}"] = band_formula(final_glm, name, cells[final_glm["band_sources"][name]])
            cells[name] = f"{col[name]}{r}"
        ws[f"{col['Frequency']}{r}"] = "=EXP(" + frequency_formula(cells, final_glm)[1:] + ")"
        ws[f"{col['Severity']}{r}"] = "=EXP(" + severity_formula(cells, final_glm)[1:] + ")"
        ws[f"{col['Pure premium']}{r}"] = f"={col['Frequency']}{r}*{col['Severity']}{r}*PPFactor*LargeLossLoad"
        ws[f"{col['Python pure premium']}{r}"] = float(p["python_price"])
        ws[f"{col['Match']}{r}"] = f"=ABS({col['Pure premium']}{r}/{col['Python pure premium']}{r}-1)<{MATCH_TOLERANCE}"
        ws[f"{col['Frequency']}{r}"].number_format = "0.00000"
        for name in ["Severity", "Pure premium", "Python pure premium"]:
            ws[f"{col[name]}{r}"].number_format = "#,##0.00"
    last = len(sample) + 1
    summary_row = last + 2
    ws[f"A{summary_row}"] = "Policies matching"
    ws[f"A{summary_row}"].font = BOLD
    ws[f"B{summary_row}"] = f"=COUNTIF({col['Match']}2:{col['Match']}{last},TRUE)&\" of \"&COUNTA({col['IDpol']}2:{col['IDpol']}{last})"
    ws.freeze_panes = "B2"
    for letter in col.values():
        ws.column_dimensions[letter].width = 13


def build_workbook(final_glm, sample):
    wb = Workbook()
    wb.remove(wb.active)
    build_readme(wb, final_glm)
    build_calculator(wb, final_glm, sample.iloc[0])
    build_factors(wb, final_glm)
    build_bands(wb, final_glm)
    build_scalars(wb, final_glm)
    build_reconcile(wb, final_glm, sample)
    return wb


def holdout_sample(predictions, final_glm, n=N_POLICIES):
    """n holdout policies (fixed seed) with Python's price: the pipeline's prediction times the large-loss load."""
    h = predictions[predictions["is_holdout"]].sample(n, random_state=SEED).copy()
    h["python_price"] = h["pp_glm_final"] * final_glm["large_loss_load"]
    return h


def recalculate_in_excel(path):
    """Open the workbook in Excel (Windows only), recalculate everything and save, so the file stores computed values."""
    script = ("$x = New-Object -ComObject Excel.Application; $x.DisplayAlerts = $false; "
              f"$w = $x.Workbooks.Open('{path}'); $x.CalculateFull(); $w.Save(); $w.Close(); $x.Quit()")
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)


def main():
    final_glm = json.loads((OUTPUTS / "final_glm.json").read_text(encoding="utf-8"))
    sample = holdout_sample(pd.read_parquet(PREDICTIONS_PATH), final_glm)
    check = python_price(final_glm, sample)
    if not np.allclose(check, sample["python_price"], rtol=1e-12):
        raise SystemExit("the rating tables do not reproduce the pipeline's prices")
    build_workbook(final_glm, sample).save(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")
    if len(sys.argv) > 1 and sys.argv[1] == "recalc":
        recalculate_in_excel(OUTPUT_PATH)
        print("recalculated in Excel and saved")


if __name__ == "__main__":
    main()
