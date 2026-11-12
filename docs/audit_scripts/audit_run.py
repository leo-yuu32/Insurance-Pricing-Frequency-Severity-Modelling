"""Phase 0 audit checks. Read-only with respect to the repo: imports src/ modules,
never calls the plotting functions that save into plots/."""
import json
import pickle
import tempfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
SCRATCH = Path(tempfile.gettempdir()) / "mtpl_audit_cache"  # cache outside the repo
SCRATCH.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "src"))

import config  # noqa: E402
import data_preprocessing  # noqa: E402
import pricing_engine  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


# ----------------------------------------------------------------------------------------
# Run (or load cached) pipeline
# ----------------------------------------------------------------------------------------
cache = SCRATCH / "pipeline_result.pkl"
raw_cache = SCRATCH / "raw.pkl"
if cache.exists() and raw_cache.exists():
    result = pickle.loads(cache.read_bytes())
    df_f, df_s = pickle.loads(raw_cache.read_bytes())
else:
    df_f, df_s = data_preprocessing.fetch_raw_data()
    raw_cache.write_bytes(pickle.dumps((df_f, df_s)))
    result = pricing_engine.run_pricing_pipeline(return_train_data=True)
    cache.write_bytes(pickle.dumps(result))

test = result["test_results"].copy()
train = result["train_data"].copy()
models = result["models"]

# ----------------------------------------------------------------------------------------
hdr("A. Reproduce scalars.json")
scalars = json.loads((REPO / "artifacts" / "scalars.json").read_text())
for k in ["off_balance_factor", "gini_technical", "gini_commercial", "avg_premium"]:
    print(f"{k:22s} file={scalars[k]:.10f}  rerun={float(result[k]):.10f}  diff={float(result[k]) - scalars[k]:+.2e}")

# ----------------------------------------------------------------------------------------
hdr("B. Data reconciliation (raw OpenML frames)")
f = df_f.copy()
s = df_s.copy()
f["ClaimNb"] = pd.to_numeric(f["ClaimNb"])
print("freq rows:", len(f), " unique IDpol:", f["IDpol"].nunique())
print("freq columns:", list(f.columns))
print("sev rows:", len(s), " unique IDpol:", s["IDpol"].nunique(), " total ClaimAmount:", f"{s['ClaimAmount'].sum():,.0f}")
print("sum ClaimNb in freq:", int(f["ClaimNb"].sum()), " policies with ClaimNb>0:", int((f["ClaimNb"] > 0).sum()))
print("ClaimNb value counts:\n", f["ClaimNb"].value_counts().sort_index().to_string())

sev_cnt = s.groupby("IDpol").agg(n_sev=("ClaimAmount", "size"), amt=("ClaimAmount", "sum")).reset_index()
orphan = sev_cnt[~sev_cnt["IDpol"].isin(f["IDpol"])]
print("\nsev IDpol not in freq: policies", len(orphan), " rows", int(orphan["n_sev"].sum()), " amount", f"{orphan['amt'].sum():,.0f}",
      f"({orphan['amt'].sum() / s['ClaimAmount'].sum():.3%} of sev amount)")

m = f.merge(sev_cnt, on="IDpol", how="left")
m["n_sev"] = m["n_sev"].fillna(0).astype(int)
m["amt"] = m["amt"].fillna(0.0)
claim_pol = m[m["ClaimNb"] > 0]
unmatched = claim_pol[claim_pol["n_sev"] == 0]
print("\npolicies ClaimNb>0:", len(claim_pol), " with NO severity record:", len(unmatched),
      f"({len(unmatched) / len(claim_pol):.2%})", " claims on them:", int(unmatched["ClaimNb"].sum()),
      f"({unmatched['ClaimNb'].sum() / f['ClaimNb'].sum():.2%} of all claims)")
print("policies ClaimNb==0 but severity rows exist:", int(((m["ClaimNb"] == 0) & (m["n_sev"] > 0)).sum()))
matched = claim_pol[claim_pol["n_sev"] > 0]
print("matched claim policies:", len(matched), " n_sev == ClaimNb:", int((matched["n_sev"] == matched["ClaimNb"]).sum()),
      " n_sev > ClaimNb:", int((matched["n_sev"] > matched["ClaimNb"]).sum()),
      " n_sev < ClaimNb:", int((matched["n_sev"] < matched["ClaimNb"]).sum()))
print("sum ClaimNb on matched:", int(matched["ClaimNb"].sum()), " sum sev rows on matched:", int(matched["n_sev"].sum()))
print("zero-amount severity rows:", int((s["ClaimAmount"] <= 0).sum()))
print("\nIDpol clustering of unmatched claim policies:")
print("  IDpol quantiles (unmatched):", unmatched["IDpol"].quantile([0, .25, .5, .75, .99, 1]).round(0).tolist())
print("  IDpol quantiles (matched):  ", matched["IDpol"].quantile([0, .01, .25, .5, .75, 1]).round(0).tolist())
for thr in [24500, 30000, 50000]:
    a = claim_pol[claim_pol["IDpol"] <= thr]
    b = claim_pol[claim_pol["IDpol"] > thr]
    print(f"  IDpol<= {thr}: claim policies {len(a)}, unmatched {int((a['n_sev'] == 0).sum())} | "
          f"IDpol> {thr}: claim policies {len(b)}, unmatched {int((b['n_sev'] == 0).sum())}")
print("  all policies with IDpol<=24500:", int((f["IDpol"] <= 24500).sum()))

print("\nFilters (applied in this order by preprocess_data):")
n0 = len(f)
low = f["Exposure"] <= config.MIN_EXPOSURE
print("  Exposure <= 0.003 dropped:", int(low.sum()), " claims on them:", int(f.loc[low, "ClaimNb"].sum()))
print("  Exposure > 1 clipped:", int((f["Exposure"] > 1).sum()), " max exposure:", float(f["Exposure"].max()))
big = f["ClaimNb"] >= config.MAX_CLAIM_NB
print("  ClaimNb >= 10 dropped:", int(big.sum()), " their ClaimNb:", f.loc[big, "ClaimNb"].tolist(),
      " exposure:", f.loc[big, "Exposure"].round(3).tolist())
print("  rows after filters:", len(train) + len(test), " (raw", n0, ")")
print("  min exposure quantiles raw:", f["Exposure"].quantile([0, .001, .01, .05, .25, .5]).round(4).tolist())

for name, d in [("train", train), ("test", test)]:
    print(f"\n{name}: rows {len(d)}, exposure {d['Exposure'].sum():,.1f}, claims {int(d['ClaimNb'].sum())}, "
          f"freq {d['ClaimNb'].sum() / d['Exposure'].sum():.5f}, policies with claim {(d['ClaimNb'] > 0).mean():.4%}, "
          f"total loss {d['TotalLoss'].sum():,.0f}, loss cost {d['TotalLoss'].sum() / d['Exposure'].sum():.2f}, "
          f"claim policies with zero loss {int(((d['ClaimNb'] > 0) & (d['TotalLoss'] == 0)).sum())}")

print("\nArea vs Density (full raw data):")
area_code = f["Area"].astype(str).map({c: i for i, c in enumerate("ABCDEF", start=1)})
dens = pd.to_numeric(f["Density"])
print("  Pearson corr(Area code, log Density):", round(float(np.corrcoef(area_code, np.log(dens))[0, 1]), 4))
print(pd.DataFrame({"Area": f["Area"].astype(str), "Density": dens}).groupby("Area")["Density"].agg(["min", "max", "count"]).to_string())

print("\nExposure by level, current model factors (train):")
for c in ["DriverAge_Bin", "VehAge_Bin", "VehPower_Bin", "VehBrand_Bin"]:
    print(train.groupby(c, observed=False)["Exposure"].sum().round(0).to_dict())
print("VehBrand raw exposure (train):", train.groupby("VehBrand")["Exposure"].sum().round(0).sort_values(ascending=False).to_dict())
print("VehPower raw exposure (train):", train.groupby("VehPower")["Exposure"].sum().round(0).to_dict())
print("BonusMalus: share ==50:", round(float((train["BonusMalus"] == 50).mean()), 4), " share >100:", round(float((train["BonusMalus"] > 100).mean()), 4),
      " share >150:", round(float((train["BonusMalus"] > 150).mean()), 5), " max:", int(train["BonusMalus"].max()))

# ----------------------------------------------------------------------------------------
hdr("C. Gini on the test set: existing function vs corrected ordering")


def gini_existing(df, pred_col, actual_col="TotalLoss"):
    return pricing_engine.calculate_gini(df, actual_col=actual_col, pred_col=pred_col, exposure_col="Exposure")


def gini_corrected(df, pred_col, actual_col="TotalLoss"):
    """Sort by predicted rate (not rate / exposure). Ties are pooled, so a constant prediction gives 0."""
    g = df.groupby(pred_col, sort=True).agg(e=("Exposure", "sum"), a=(actual_col, "sum"))
    x = np.concatenate(([0.0], g["e"].cumsum().values / g["e"].sum()))
    y = np.concatenate(([0.0], g["a"].cumsum().values / g["a"].sum()))
    return 1 - 2 * np.trapezoid(y, x)


test["Const"] = 1.0
only_1821 = {"DriverAge_Bin": {"18-21": config.ACTUARIAL_LOADINGS["DriverAge_Bin"]["18-21"]}}
test["Cal_x_1821_only"] = np.where(test["DriverAge_Bin"].astype(str) == "18-21",
                                   test["PurePremium_Calibrated"] * 2.40, test["PurePremium_Calibrated"])
age_only = test["PurePremium_Calibrated"] * test["DriverAge_Bin"].astype(str).map(config.ACTUARIAL_LOADINGS["DriverAge_Bin"]).fillna(1.0)
test["Cal_x_both_age_loadings"] = age_only
test["InvExposure"] = 1.0  # placeholder; existing fn on Const already sorts by 1/Exposure

rows = []
for label, col in [("Constant prediction", "Const"), ("Raw (freq x sev)", "PurePremium_Raw"),
                   ("Off-balanced [CV 0.43]", "PurePremium_OB"), ("Isotonic calibrated", "PurePremium_Calibrated"),
                   ("Calibrated x 18-21 loading only", "Cal_x_1821_only"),
                   ("Calibrated x both age loadings", "Cal_x_both_age_loadings"),
                   ("Final price (all 7) [CV 0.48]", "Final_Price")]:
    rows.append({"premium": label, "existing_fn": gini_existing(test, col), "corrected": gini_corrected(test, col)})
print(pd.DataFrame(rows).to_string(index=False))
print("\nFrequency Gini (actual = ClaimNb, pred = PredFreq):  existing_fn", round(gini_existing(test, "PredFreq", "ClaimNb"), 4),
      " corrected", round(gini_corrected(test, "PredFreq", "ClaimNb"), 4))
print("Constant prediction, actual = ClaimNb: existing_fn", round(gini_existing(test, "Const", "ClaimNb"), 4))
print("distinct values: OB", test["PurePremium_OB"].nunique(), " calibrated", test["PurePremium_Calibrated"].nunique(),
      " final", test["Final_Price"].nunique())
print("\nClaim frequency and loss cost by exposure band (test):")
eb = pd.cut(test["Exposure"], [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.999, 1.0])
t = test.groupby(eb, observed=False).agg(policies=("Exposure", "size"), exposure=("Exposure", "sum"), claims=("ClaimNb", "sum"), loss=("TotalLoss", "sum"))
t["freq"] = t["claims"] / t["exposure"]
t["loss_cost"] = t["loss"] / t["exposure"]
t["pred_OB_rate"] = test.assign(x=test["PurePremium_OB"] * test["Exposure"]).groupby(eb, observed=False)["x"].sum() / t["exposure"]
print(t.to_string())

# ----------------------------------------------------------------------------------------
hdr("D. Actual versus expected on the test set, by stage")
cap15 = config.CALIBRATION_CAP_THRESHOLD
test["Loss_cap15k"] = test["TotalLoss"].clip(upper=cap15)
stages = ["PurePremium_Raw", "PurePremium_OB", "PurePremium_Calibrated", "Final_Price"]
rows = []
for sname in stages:
    exp = (test[sname] * test["Exposure"]).sum()
    rows.append({"stage": sname, "expected": exp, "A/E uncapped": test["TotalLoss"].sum() / exp, "A/E capped15k": test["Loss_cap15k"].sum() / exp})
print(pd.DataFrame(rows).to_string(index=False))
print("test actual uncapped:", f"{test['TotalLoss'].sum():,.0f}", " capped15k:", f"{test['Loss_cap15k'].sum():,.0f}")
print("frequency A/E test:", round(test["ClaimNb"].sum() / (test["PredFreq"] * test["Exposure"]).sum(), 4),
      " train:", round(train["ClaimNb"].sum() / (train["PredFreq"] * train["Exposure"]).sum(), 4))
zero = test["PurePremium_Calibrated"] == 0
print("calibrated premium == 0: policies", int(zero.sum()), f"({zero.mean():.2%})", " exposure share", f"{test.loc[zero, 'Exposure'].sum() / test['Exposure'].sum():.2%}")
print("calibrated premium range:", round(float(test["PurePremium_Calibrated"].min()), 2), "to", round(float(test["PurePremium_Calibrated"].max()), 2),
      "| OB range:", round(float(test["PurePremium_OB"].min()), 2), "to", round(float(test["PurePremium_OB"].max()), 2))

for name, d in [("TEST", test), ("TRAIN", train)]:
    d = d.copy()
    d["Loss_cap15k"] = d["TotalLoss"].clip(upper=cap15)
    g = d.groupby("DriverAge_Bin", observed=False)
    out = pd.DataFrame({"exposure": g["Exposure"].sum(), "claims": g["ClaimNb"].sum()})
    out["act_freq"] = out["claims"] / out["exposure"]
    out["pred_freq"] = g.apply(lambda x: (x["PredFreq"] * x["Exposure"]).sum(), include_groups=False) / out["exposure"]
    out["freq A/E"] = out["act_freq"] / out["pred_freq"]
    out["act_loss_cost"] = g["TotalLoss"].sum() / out["exposure"]
    out["act_lc_cap15k"] = g["Loss_cap15k"].sum() / out["exposure"]
    cols = ["PurePremium_OB"] + (["PurePremium_Calibrated", "Final_Price"] if name == "TEST" else [])
    for c in cols:
        out[c] = g.apply(lambda x, c=c: (x[c] * x["Exposure"]).sum(), include_groups=False) / out["exposure"]
        out[f"A/E {c[-10:]}"] = out["act_loss_cost"] / out[c]
    print(f"\n{name} by driver-age band:")
    print(out.to_string())

freq_coefs = pd.read_csv(REPO / "artifacts" / "frequency_coefficients.csv")
print("\nFitted frequency relativities for driver age (reference 18-21 = 1.00):")
print(freq_coefs[freq_coefs["encoded_feature"].str.startswith("DriverAge")].assign(rel=lambda x: np.exp(x["coefficient"])).to_string(index=False))

# ----------------------------------------------------------------------------------------
hdr("E. Baseline frequency metrics of the current model (test set)")


def poisson_dev(y, mu):
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    t1 = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2 * np.sum(t1 - (y - mu))


mu_te = test["PredFreq"] * test["Exposure"]
d_model = poisson_dev(test["ClaimNb"], mu_te)
null_rate = train["ClaimNb"].sum() / train["Exposure"].sum()
d_null = poisson_dev(test["ClaimNb"], null_rate * test["Exposure"])
print(f"model: total deviance {d_model:,.1f} | per policy {d_model / len(test):.6f} | per unit exposure {d_model / test['Exposure'].sum():.6f}")
print(f"null (train mean frequency {null_rate:.5f}): total {d_null:,.1f} | per policy {d_null / len(test):.6f} | per unit exposure {d_null / test['Exposure'].sum():.6f}")
print(f"deviance explained on test: {1 - d_model / d_null:.4%}")
from sklearn.metrics import mean_poisson_deviance  # noqa: E402
print("sklearn mean_poisson_deviance (rate, exposure-weighted):",
      round(mean_poisson_deviance(test["ClaimNb"] / test["Exposure"], test["PredFreq"], sample_weight=test["Exposure"]), 6))

# ----------------------------------------------------------------------------------------
hdr("F. Caps")
nz_tr = train[(train["ClaimNb"] > 0) & (train["AvgSeverity"] > 0)]
full = pd.concat([train, test])
nz_full = full[(full["ClaimNb"] > 0) & (full["AvgSeverity"] > 0)]
cap_tr = nz_tr["AvgSeverity"].quantile(config.SEVERITY_CAP_PERCENTILE)
cap_full = nz_full["AvgSeverity"].quantile(config.SEVERITY_CAP_PERCENTILE)
print(f"severity training rows (claim policies with loss>0): {len(nz_tr)}; claim policies in train: {int((train['ClaimNb'] > 0).sum())}")
print(f"99.9th pct of policy-level AvgSeverity: train {cap_tr:,.2f} | full data {cap_full:,.2f}")
capped = nz_tr[nz_tr["AvgSeverity"] > cap_tr]
removed = ((capped["AvgSeverity"] - cap_tr) * capped["ClaimNb"]).sum()
print(f"train rows capped: {len(capped)}; loss removed {removed:,.0f} = {removed / train['TotalLoss'].sum():.2%} of train loss")
print(f"99.9th pct of individual claim amounts (raw sev file): {df_s['ClaimAmount'].quantile(0.999):,.2f}; max claim {df_s['ClaimAmount'].max():,.0f}")
for name, d in [("train", train), ("test", test)]:
    over = d["TotalLoss"] > cap15
    rem = (d.loc[over, "TotalLoss"] - cap15).sum()
    top10 = d["TotalLoss"].nlargest(10).sum()
    print(f"{name}: policies with loss > 15k: {int(over.sum())} ({over.sum() / (d['TotalLoss'] > 0).sum():.2%} of loss-making policies); "
          f"15k cap removes {rem:,.0f} = {rem / d['TotalLoss'].sum():.2%} of loss; largest policy {d['TotalLoss'].max():,.0f} "
          f"({d['TotalLoss'].max() / d['TotalLoss'].sum():.2%}); top-10 policies {top10 / d['TotalLoss'].sum():.2%} of loss")
print("AvgSeverity quantiles (train claim policies):", nz_tr["AvgSeverity"].quantile([.5, .9, .99, .999]).round(0).tolist())
knots = pd.read_csv(REPO / "artifacts" / "isotonic_knots.csv")
print("isotonic knots in artifact:", len(knots), " distinct y levels:", knots["y"].nunique(), " lowest y:", knots["y"].min(), " highest y:", round(knots["y"].max(), 2))
cal = models["calibrator"]
print("isotonic knots in rerun:", len(cal.X_thresholds_), " distinct y:", len(np.unique(cal.y_thresholds_)))

# ----------------------------------------------------------------------------------------
hdr("G. Excel workbook Match column (cached values only)")
from openpyxl import load_workbook  # noqa: E402

wb = load_workbook(REPO / "artifacts" / "rating_engine.xlsx", data_only=True, read_only=True)
ws = wb["Reconcile"]
rows_x = list(ws.iter_rows(values_only=True))
header = list(rows_x[0])
mi = header.index("Match")
pi = header.index("Pct_Diff")
vals = [r[mi] for r in rows_x[1:101]]
pct = [r[pi] for r in rows_x[1:101]]
print("sheets:", wb.sheetnames)
print("Match cached values:", pd.Series(vals).value_counts(dropna=False).to_dict())
pct_num = [abs(p) for p in pct if isinstance(p, (int, float))]
print("Pct_Diff cached: n numeric", len(pct_num), " max abs", (max(pct_num) if pct_num else None))
for r in rows_x[101:107]:
    if r and r[0] is not None:
        print("  summary row:", r[:2])
samp = pd.read_csv(REPO / "artifacts" / "test_sample.csv")
j = samp.merge(test[["Final_Price"]], left_on="original_index", right_index=True, suffixes=("_file", "_rerun"))
print("test_sample.csv vs rerun Final_Price: rows matched", len(j), " max abs diff", float((j["Final_Price_file"] - j["Final_Price_rerun"]).abs().max()))
py_idx = header.index("Py_Final_Price")
xl_py = [r[py_idx] for r in rows_x[1:101]]
print("workbook Py_Final_Price vs test_sample.csv max abs diff:", float(np.max(np.abs(np.array(xl_py, float) - samp["Final_Price"].values))))
