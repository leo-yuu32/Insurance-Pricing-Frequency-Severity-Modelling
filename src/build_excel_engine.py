"""
build_excel_engine.py

Phase 2 of the Excel cross-validation workstream.

Reads artifacts/ (produced by Phase 1) and writes artifacts/rating_engine.xlsx
containing a working Excel replica of the pricing pipeline:

    Sheets
    ------
    README          Index of what each sheet does
    Scalars         Off-balance factor and key hyperparameters
    Coef_Freq       Poisson GLM coefficients (intercept + dummies)
    Coef_Sev        Gamma GLM coefficients
    Encoder_Map     Reference levels per categorical feature
    Knots           Isotonic calibration breakpoints (x, y)
    Loadings        Editable commercial multipliers (the sensitivity input)
    Reconcile       100 test policies, Excel vs Python predictions, % diff

Run from project root:
    python src/build_excel_engine.py
"""

import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.workbook.defined_name import DefinedName

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
OUTPUT_PATH = ARTIFACTS_DIR / "rating_engine.xlsx"

# Industry-standard color coding for financial models
BLUE = Font(color="0000FF", name="Arial", size=10)        # Inputs / scenarios
BLACK = Font(color="000000", name="Arial", size=10)       # Formulas
GREEN = Font(color="008000", name="Arial", size=10)       # Cross-sheet links
BOLD = Font(bold=True, name="Arial", size=10)
HEADER = Font(bold=True, color="FFFFFF", name="Arial", size=10)
HEADER_FILL = PatternFill("solid", start_color="305496")  # Dark blue
EDITABLE_FILL = PatternFill("solid", start_color="FFF2CC")  # Light yellow
PASS_FILL = PatternFill("solid", start_color="C6EFCE")
FAIL_FILL = PatternFill("solid", start_color="FFC7CE")
THIN = Side(border_style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def load_artifacts(artifacts_dir):
    """Load every CSV/JSON Phase 1 produces."""
    a = {}
    a["freq_coefs"] = pd.read_csv(artifacts_dir / "frequency_coefficients.csv")
    a["sev_coefs"] = pd.read_csv(artifacts_dir / "severity_coefficients.csv")
    a["ref_levels"] = pd.read_csv(artifacts_dir / "encoder_reference_levels.csv")
    a["knots"] = pd.read_csv(artifacts_dir / "isotonic_knots.csv")
    a["loadings"] = pd.read_csv(artifacts_dir / "loadings.csv")
    a["test_sample"] = pd.read_csv(artifacts_dir / "test_sample.csv")
    with open(artifacts_dir / "scalars.json") as f:
        a["scalars"] = json.load(f)
    return a


def style_header_row(ws, row, n_cols):
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = HEADER
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = BORDER


def add_defined_name(wb, name, ref):
    """Workbook-scoped named range."""
    wb.defined_names[name] = DefinedName(name=name, attr_text=ref)


# -----------------------------------------------------------------------------
# Individual sheet builders
# -----------------------------------------------------------------------------

def build_readme(wb):
    ws = wb.create_sheet("README", 0)
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 90

    ws["A1"] = "Insurance Pricing Engine - Excel Cross-Validation"
    ws["A1"].font = Font(bold=True, size=14, name="Arial")
    ws.merge_cells("A1:B1")

    rows = [
        ("", ""),
        ("Purpose", "Replicates the Python pricing pipeline in Excel to validate model outputs "
                    "and stress-test commercial loadings."),
        ("", ""),
        ("Sheet", "Description"),
        ("Scalars", "Off-balance factor and key hyperparameters (read-only reference)."),
        ("Coef_Freq", "Poisson GLM intercept and dummy-variable coefficients."),
        ("Coef_Sev", "Gamma GLM intercept and dummy-variable coefficients."),
        ("Encoder_Map", "Reference level dropped by OneHotEncoder per categorical feature. "
                        "A policy whose value equals the reference contributes 0 to the sum."),
        ("Knots", "Isotonic calibration breakpoints. Excel interpolates linearly between them."),
        ("Loadings", "Commercial loading multipliers. EDIT THESE (yellow cells) to flex pricing "
                     "and watch Reconcile!Excel_Final_Price respond."),
        ("Reconcile", "100 test policies. Compares Excel-computed Final Price to Python's, "
                      "with % difference and a Match flag. Successful reconciliation = all "
                      "diffs under ~0.01% (floating-point precision)."),
        ("", ""),
        ("Color key", "Blue = input you can edit | Black = formula | Green = cross-sheet link "
                      "| Yellow fill = editable loading"),
        ("", ""),
        ("Regenerate",
         "If the underlying Python model changes, rerun: "
         "python src/export_model_artifacts.py && python src/build_excel_engine.py"),
    ]
    for i, (a, b) in enumerate(rows, start=2):
        ws.cell(row=i, column=1, value=a).font = BOLD if a in (
            "Purpose", "Sheet", "Color key", "Regenerate") else Font(name="Arial", size=10)
        ws.cell(row=i, column=2, value=b).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[i].height = 30 if len(b) > 90 else 18


def build_scalars(wb, scalars):
    ws = wb.create_sheet("Scalars")
    ws["A1"] = "Key"
    ws["B1"] = "Value"
    style_header_row(ws, 1, 2)
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 16

    keys = list(scalars.keys())
    for i, k in enumerate(keys, start=2):
        ws.cell(row=i, column=1, value=k).font = BOLD
        v_cell = ws.cell(row=i, column=2, value=scalars[k])
        v_cell.font = BLUE  # values that could in principle be flexed
        if isinstance(scalars[k], float):
            v_cell.number_format = "0.000000"

    # Named range for the off-balance factor (used in Reconcile formulas)
    ob_row = keys.index("off_balance_factor") + 2
    add_defined_name(wb, "OffBalance", f"Scalars!$B${ob_row}")


def build_coef_sheet(wb, sheet_name, coefs, intercept_name, table_name):
    ws = wb.create_sheet(sheet_name)
    ws["A1"] = "encoded_feature"
    ws["B1"] = "coefficient"
    style_header_row(ws, 1, 2)
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 16

    intercept_row = None
    for i, row in coefs.iterrows():
        r = i + 2
        ws.cell(row=r, column=1, value=row["encoded_feature"])
        c = ws.cell(row=r, column=2, value=float(row["coefficient"]))
        c.number_format = "0.000000"
        if row["encoded_feature"] == "_intercept_":
            intercept_row = r
            ws.cell(row=r, column=1).font = BOLD
            ws.cell(row=r, column=2).font = BOLD

    last_row = len(coefs) + 1
    add_defined_name(wb, intercept_name, f"{sheet_name}!$B${intercept_row}")
    add_defined_name(wb, table_name, f"{sheet_name}!$A$2:$B${last_row}")


def build_encoder_map(wb, ref_levels):
    ws = wb.create_sheet("Encoder_Map")
    ws["A1"] = "feature"
    ws["B1"] = "reference_level"
    ws["C1"] = "non_reference_levels"
    style_header_row(ws, 1, 3)
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 60

    for i, row in ref_levels.iterrows():
        r = i + 2
        ws.cell(row=r, column=1, value=row["feature"]).font = BOLD
        ws.cell(row=r, column=2, value=row["reference_level"])
        ws.cell(row=r, column=3, value=row["non_reference_levels"])


def build_knots(wb, knots):
    ws = wb.create_sheet("Knots")
    ws["A1"] = "x"
    ws["B1"] = "y"
    style_header_row(ws, 1, 2)
    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 14

    for i, row in knots.iterrows():
        r = i + 2
        ws.cell(row=r, column=1, value=float(row["x"])).number_format = "0.0000"
        ws.cell(row=r, column=2, value=float(row["y"])).number_format = "0.0000"

    n = len(knots)
    add_defined_name(wb, "KnotsX", f"Knots!$A$2:$A${n + 1}")
    add_defined_name(wb, "KnotsY", f"Knots!$B$2:$B${n + 1}")


def build_loadings(wb, loadings):
    ws = wb.create_sheet("Loadings")
    ws["A1"] = "feature"
    ws["B1"] = "bin"
    ws["C1"] = "multiplier"
    ws["D1"] = "key (helper)"
    style_header_row(ws, 1, 4)
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 28

    for i, row in loadings.iterrows():
        r = i + 2
        ws.cell(row=r, column=1, value=row["feature"]).font = BOLD
        ws.cell(row=r, column=2, value=str(row["bin"]))
        # Editable multiplier (yellow fill, blue text = input cell)
        c = ws.cell(row=r, column=3, value=float(row["multiplier"]))
        c.font = BLUE
        c.fill = EDITABLE_FILL
        c.number_format = "0.0000"
        # Helper key column: feature|bin (formula so it follows edits)
        ws.cell(row=r, column=4, value=f"=A{r}&\"|\"&B{r}")

    n = len(loadings)
    # Two-column lookup table: key, multiplier (cols D:C are wrong order, fix below)
    # We need key in col 1, multiplier in col 2 for VLOOKUP. Use D and then reference C.
    # Simpler: use INDEX/MATCH with the helper key.
    add_defined_name(wb, "LoadingsKeys", f"Loadings!$D$2:$D${n + 1}")
    add_defined_name(wb, "LoadingsMult", f"Loadings!$C$2:$C${n + 1}")


# -----------------------------------------------------------------------------
# The reconciliation sheet - where everything comes together
# -----------------------------------------------------------------------------

# Categorical features in fixed order (must match config.CATEGORICAL_FEATURES)
CAT_FEATURES = [
    "VehBrand_Bin", "VehGas", "Region", "Area",
    "VehPower_Bin", "VehAge_Bin", "DriverAge_Bin",
]

# Features that have loadings applied (subset of CAT_FEATURES, must match config)
LOADING_FEATURES = ["DriverAge_Bin", "VehAge_Bin", "VehPower_Bin"]


def build_reconcile(wb, test_sample):
    ws = wb.create_sheet("Reconcile")

    # Column layout
    cols = []
    # Identifiers and raw features (informational)
    cols += ["original_index", "DriverAge", "VehAge", "VehPower", "VehBrand"]
    # Binned features the model consumes
    cols += CAT_FEATURES
    # Exposure and ground truth
    cols += ["Exposure", "ClaimNb", "TotalLoss"]
    # Excel-computed predictions at each pipeline stage
    cols += ["Excel_PredFreq", "Excel_PredSev",
             "Excel_PurePremium_Raw", "Excel_PurePremium_OB",
             "Excel_PurePremium_Calibrated",
             "Excel_LoadingMult", "Excel_Final_Price"]
    # Python's predictions and the deltas
    cols += ["Py_Final_Price", "Diff", "Pct_Diff", "Match"]

    # Header row
    for j, name in enumerate(cols, start=1):
        ws.cell(row=1, column=j, value=name)
    style_header_row(ws, 1, len(cols))
    ws.freeze_panes = "A2"

    # Column index lookup
    cidx = {name: i + 1 for i, name in enumerate(cols)}

    # Width hints
    for name, w in [("original_index", 8), ("DriverAge", 8), ("VehAge", 7),
                    ("VehPower", 8), ("VehBrand", 9), ("VehBrand_Bin", 11),
                    ("VehGas", 8), ("Region", 7), ("Area", 6),
                    ("VehPower_Bin", 11), ("VehAge_Bin", 11), ("DriverAge_Bin", 12),
                    ("Exposure", 9), ("ClaimNb", 8), ("TotalLoss", 11),
                    ("Excel_PredFreq", 13), ("Excel_PredSev", 13),
                    ("Excel_PurePremium_Raw", 16), ("Excel_PurePremium_OB", 16),
                    ("Excel_PurePremium_Calibrated", 22),
                    ("Excel_LoadingMult", 14), ("Excel_Final_Price", 14),
                    ("Py_Final_Price", 14), ("Diff", 10), ("Pct_Diff", 10),
                    ("Match", 7)]:
        ws.column_dimensions[get_column_letter(cidx[name])].width = w

    # ---------- Populate rows ----------
    for i, (_, sample_row) in enumerate(test_sample.iterrows()):
        r = i + 2  # Excel row

        # Raw inputs (just copy values)
        ws.cell(r, cidx["original_index"], int(sample_row["original_index"]))
        ws.cell(r, cidx["DriverAge"], int(sample_row["DriverAge"]))
        ws.cell(r, cidx["VehAge"], int(sample_row["VehAge"]))
        ws.cell(r, cidx["VehPower"], int(sample_row["VehPower"]))
        ws.cell(r, cidx["VehBrand"], str(sample_row["VehBrand"]))
        for f in CAT_FEATURES:
            ws.cell(r, cidx[f], str(sample_row[f]))
        ws.cell(r, cidx["Exposure"], float(sample_row["Exposure"])).number_format = "0.0000"
        ws.cell(r, cidx["ClaimNb"], int(sample_row["ClaimNb"]))
        ws.cell(r, cidx["TotalLoss"], float(sample_row["TotalLoss"])).number_format = "0.00"

        # ---------- Frequency: intercept + sum of dummy contributions ----------
        # For each categorical feature, build the encoded name (feature + "_" + value)
        # and VLOOKUP in the freq coefficient table. IFERROR -> 0 covers reference levels
        # (which aren't in the table because OneHotEncoder dropped them).
        freq_lookups = " + ".join(
            f'IFERROR(VLOOKUP("{f}_"&{get_column_letter(cidx[f])}{r}, FreqCoefs, 2, FALSE), 0)'
            for f in CAT_FEATURES
        )
        ws.cell(r, cidx["Excel_PredFreq"],
                f"=EXP(FreqIntercept + {freq_lookups})").number_format = "0.000000"

        # ---------- Severity: same shape ----------
        sev_lookups = " + ".join(
            f'IFERROR(VLOOKUP("{f}_"&{get_column_letter(cidx[f])}{r}, SevCoefs, 2, FALSE), 0)'
            for f in CAT_FEATURES
        )
        ws.cell(r, cidx["Excel_PredSev"],
                f"=EXP(SevIntercept + {sev_lookups})").number_format = "0.00"

        pf_col = get_column_letter(cidx["Excel_PredFreq"])
        ps_col = get_column_letter(cidx["Excel_PredSev"])
        pp_raw_col = get_column_letter(cidx["Excel_PurePremium_Raw"])
        pp_ob_col = get_column_letter(cidx["Excel_PurePremium_OB"])
        pp_cal_col = get_column_letter(cidx["Excel_PurePremium_Calibrated"])
        lm_col = get_column_letter(cidx["Excel_LoadingMult"])
        fp_col = get_column_letter(cidx["Excel_Final_Price"])
        py_col = get_column_letter(cidx["Py_Final_Price"])
        d_col = get_column_letter(cidx["Diff"])
        pd_col = get_column_letter(cidx["Pct_Diff"])

        ws.cell(r, cidx["Excel_PurePremium_Raw"],
                f"={pf_col}{r}*{ps_col}{r}").number_format = "0.00"
        ws.cell(r, cidx["Excel_PurePremium_OB"],
                f"={pp_raw_col}{r}*OffBalance").number_format = "0.00"

        # ---------- Isotonic calibration: piecewise linear (no LET, broadly portable) ----------
        # MATCH(v, KnotsX, 1) finds the position of the largest knot <= v.
        # Clip outside the range manually with the outer IFs.
        # We repeat MATCH/INDEX a few times to avoid LET (not in older Excel/Calc),
        # but the cost is negligible - only ~hundreds of knots and 100 rows here.
        iso = (
            f"=IF({pp_ob_col}{r}<=MIN(KnotsX),INDEX(KnotsY,1),"
            f"IF({pp_ob_col}{r}>=MAX(KnotsX),INDEX(KnotsY,COUNT(KnotsX)),"
            f"INDEX(KnotsY,MATCH({pp_ob_col}{r},KnotsX,1))+"
            f"({pp_ob_col}{r}-INDEX(KnotsX,MATCH({pp_ob_col}{r},KnotsX,1)))*"
            f"(INDEX(KnotsY,MATCH({pp_ob_col}{r},KnotsX,1)+1)-INDEX(KnotsY,MATCH({pp_ob_col}{r},KnotsX,1)))/"
            f"(INDEX(KnotsX,MATCH({pp_ob_col}{r},KnotsX,1)+1)-INDEX(KnotsX,MATCH({pp_ob_col}{r},KnotsX,1)))"
            f"))"
        )
        ws.cell(r, cidx["Excel_PurePremium_Calibrated"], iso).number_format = "0.00"

        # ---------- Loading multiplier: product of (feature, bin) lookups ----------
        # IFERROR -> 1 since most policies don't trigger any loading.
        load_lookups = " * ".join(
            f'IFERROR(INDEX(LoadingsMult, MATCH("{f}|"&{get_column_letter(cidx[f])}{r}, LoadingsKeys, 0)), 1)'
            for f in LOADING_FEATURES
        )
        ws.cell(r, cidx["Excel_LoadingMult"],
                f"={load_lookups}").number_format = "0.0000"

        ws.cell(r, cidx["Excel_Final_Price"],
                f"={pp_cal_col}{r}*{lm_col}{r}").number_format = "0.00"

        # ---------- Python ground truth and deltas ----------
        ws.cell(r, cidx["Py_Final_Price"],
                float(sample_row["Final_Price"])).number_format = "0.00"
        ws.cell(r, cidx["Diff"],
                f"={fp_col}{r}-{py_col}{r}").number_format = "0.0000"
        ws.cell(r, cidx["Pct_Diff"],
                f"=IFERROR({d_col}{r}/{py_col}{r},0)").number_format = "0.0000%"
        # Match tolerance: 0.01% covers float precision but flags real divergence
        ws.cell(r, cidx["Match"],
                f"=IF(ABS({pd_col}{r})<0.0001,\"PASS\",\"FAIL\")")

    # Conditional formatting on the Match column
    n_rows = len(test_sample)
    match_col = get_column_letter(cidx["Match"])
    for r in range(2, n_rows + 2):
        cell = ws.cell(r, cidx["Match"])
        # We'll set fill based on recalc later via openpyxl conditional formatting
    from openpyxl.formatting.rule import CellIsRule
    ws.conditional_formatting.add(
        f"{match_col}2:{match_col}{n_rows + 1}",
        CellIsRule(operator="equal", formula=['"PASS"'], fill=PASS_FILL),
    )
    ws.conditional_formatting.add(
        f"{match_col}2:{match_col}{n_rows + 1}",
        CellIsRule(operator="equal", formula=['"FAIL"'], fill=FAIL_FILL),
    )

    # Summary cells above the data
    ws.cell(2 + n_rows + 1, 1, "SUMMARY").font = BOLD
    ws.cell(2 + n_rows + 2, 1, "Pass count")
    ws.cell(2 + n_rows + 2, 2,
            f'=COUNTIF({match_col}2:{match_col}{n_rows + 1},"PASS")')
    # MAX(MAX, -MIN) is the array-free way to compute max absolute value
    ws.cell(2 + n_rows + 3, 1, "Max |Pct_Diff|")
    ws.cell(2 + n_rows + 3, 2,
            f"=MAX(MAX({pd_col}2:{pd_col}{n_rows + 1}),"
            f"-MIN({pd_col}2:{pd_col}{n_rows + 1}))").number_format = "0.0000%"
    # SUMPRODUCT forces array evaluation without Ctrl+Shift+Enter
    ws.cell(2 + n_rows + 4, 1, "Mean |Pct_Diff|")
    ws.cell(2 + n_rows + 4, 2,
            f"=SUMPRODUCT(ABS({pd_col}2:{pd_col}{n_rows + 1}))/{n_rows}"
            ).number_format = "0.0000%"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(artifacts_dir=ARTIFACTS_DIR, output_path=OUTPUT_PATH):
    print(f"Reading artifacts from: {artifacts_dir}/")
    a = load_artifacts(artifacts_dir)

    wb = Workbook()
    wb.remove(wb.active)  # Drop the default empty sheet

    build_readme(wb)
    build_scalars(wb, a["scalars"])
    build_coef_sheet(wb, "Coef_Freq", a["freq_coefs"], "FreqIntercept", "FreqCoefs")
    build_coef_sheet(wb, "Coef_Sev", a["sev_coefs"], "SevIntercept", "SevCoefs")
    build_encoder_map(wb, a["ref_levels"])
    build_knots(wb, a["knots"])
    build_loadings(wb, a["loadings"])
    build_reconcile(wb, a["test_sample"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)

    print(f"\nWorkbook written to: {output_path}")
    print("\nSheets in order:")
    for name in wb.sheetnames:
        print(f"  - {name}")
    print("\nDefined names:")
    for dn in wb.defined_names:
        print(f"  - {dn}")
    print("\nOpen the workbook in Excel/LibreOffice. The Reconcile sheet's Match")
    print("column should be all PASS (under 0.01% diff on every policy).")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]), Path(sys.argv[1]) / "rating_engine.xlsx")
    else:
        main()
