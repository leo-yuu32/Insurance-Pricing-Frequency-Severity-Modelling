"""Sanity check on the corrected-ordering Gini jump (0.08 -> 0.32 from one loading):
how much is one large claim? Uses the cached pipeline result only."""
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRATCH = Path(tempfile.gettempdir()) / "mtpl_audit_cache"  # cache outside the repo
SCRATCH.mkdir(exist_ok=True)
result = pickle.loads((SCRATCH / "pipeline_result.pkl").read_bytes())
test = result["test_results"].copy()
pd.set_option("display.width", 250)


def gini_corrected(df, pred_col, actual_col):
    g = df.groupby(pred_col, sort=True).agg(e=("Exposure", "sum"), a=(actual_col, "sum"))
    x = np.concatenate(([0.0], g["e"].cumsum().values / g["e"].sum()))
    y = np.concatenate(([0.0], g["a"].cumsum().values / g["a"].sum()))
    return 1 - 2 * np.trapezoid(y, x)


def gini_existing(df, pred_col, actual_col):
    d = df.copy()
    d["r"] = d[pred_col] / d["Exposure"]
    d = d.sort_values("r")
    x = np.concatenate(([0.0], d["Exposure"].cumsum().values / d["Exposure"].sum()))
    y = np.concatenate(([0.0], d[actual_col].cumsum().values / d[actual_col].sum()))
    return 1 - 2 * np.trapezoid(y, x)


test["Const"] = 1.0
test["Cal_x_1821_only"] = np.where(test["DriverAge_Bin"].astype(str) == "18-21", test["PurePremium_Calibrated"] * 2.40, test["PurePremium_Calibrated"])
test["Loss_cap15k"] = test["TotalLoss"].clip(upper=15000)

top = test.nlargest(5, "TotalLoss")[["DriverAge", "DriverAge_Bin", "VehAge_Bin", "VehPower_Bin", "Exposure", "ClaimNb", "TotalLoss", "PurePremium_OB", "PurePremium_Calibrated", "Final_Price"]]
print("Five largest test-set policies by loss:\n", top.to_string())
big_idx = test["TotalLoss"].idxmax()
ex1 = test.drop(index=big_idx)

cols = [("Constant prediction", "Const"), ("Off-balanced", "PurePremium_OB"), ("Isotonic calibrated", "PurePremium_Calibrated"),
        ("Calibrated x 18-21 loading only", "Cal_x_1821_only"), ("Final price (all 7)", "Final_Price")]
rows = []
for label, c in cols:
    rows.append({
        "premium": label,
        "corr | all losses": gini_corrected(test, c, "TotalLoss"),
        "corr | excl. largest policy": gini_corrected(ex1, c, "TotalLoss"),
        "corr | losses capped 15k": gini_corrected(test, c, "Loss_cap15k"),
        "corr | claim counts": gini_corrected(test, c, "ClaimNb"),
        "existing | all losses": gini_existing(test, c, "TotalLoss"),
        "existing | excl. largest policy": gini_existing(ex1, c, "TotalLoss"),
    })
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("\nFrequency model, corrected ordering by PredFreq: claim counts", round(gini_corrected(test, "PredFreq", "ClaimNb"), 4))
