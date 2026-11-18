import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression

import data_preprocessing
import frequency_model
import severity_model
import config

# Find project root dynamically so plots save correctly regardless of where script is run
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent.parent
PLOTS_DIR = ROOT_DIR / "plots"


def get_plots_dir():
    """
    Returns the absolute path to the plots directory.
    Creates the directory if it doesn't exist.

    Returns:
    --------
    Path: Absolute path to plots directory
    """
    if not PLOTS_DIR.exists():
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {PLOTS_DIR}/")
    return PLOTS_DIR


def train_calibrator(predictions, actuals, exposure, cap_threshold=None):
    """
    Fits an Isotonic Regression model with capped actuals.
    """
    if cap_threshold is None:
        cap_threshold = config.CALIBRATION_CAP_THRESHOLD
    iso_reg = IsotonicRegression(y_min=0, out_of_bounds="clip")

    # Cap actuals before calibration to prevent extreme losses from skewing the calibration curve
    # This ensures calibration focuses on typical claims, not catastrophic outliers
    capped_actuals = actuals.clip(upper=cap_threshold)

    # Calculate actual rate using capped losses - this is what we calibrate against
    actual_rate = np.divide(
        capped_actuals, exposure, out=np.zeros_like(capped_actuals), where=exposure != 0
    )

    # Weight by exposure so high-exposure policies have more influence on calibration
    iso_reg.fit(predictions, actual_rate, sample_weight=exposure)
    return iso_reg


def map_feature_name(feature_name):
    """
    Maps user-friendly feature names to actual dataframe column names.

    Parameters:
    -----------
    feature_name : str
        User-friendly feature name (e.g., 'Driver Age', 'Vehicle Age', 'Vehicle Power')

    Returns:
    --------
    str: Actual dataframe column name
    """
    mapping = {
        "Driver Age": "DriverAge_Bin",
        "Vehicle Age": "VehAge_Bin",
        "Vehicle Power": "VehPower_Bin",
    }
    return mapping.get(feature_name, feature_name)


def plot_one_way(df, feature, exposure_col="Exposure", filename=None, return_fig=False):
    """
    Generates a professional one-way analysis chart showing Actual vs Model vs Final Price.

    Creates a dual-axis chart with:
    - Left Y-axis: Three lines (Actual Average Loss, Model Predicted, Final Price)
    - Right Y-axis: Exposure volume bars

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    feature : str
        Feature name or column name to analyze (e.g., 'Driver Age' or 'DriverAge_Bin')
    exposure_col : str
        Column name for exposure (default: 'Exposure')
    filename : str, optional
        Filename to save the chart (if provided, saves instead of returning)
    return_fig : bool, default False
        If True, returns matplotlib figure object instead of showing/saving

    Returns:
    --------
    If return_fig=True: matplotlib.figure.Figure object
    If filename provided: None (saves file)
    Otherwise: pd.DataFrame with aggregated data
    """
    df_plot = df.copy()

    # Cap actuals for visualization - prevents single massive claim from making chart unreadable
    df_plot["TotalLoss_Visual"] = df_plot["TotalLoss"].clip(upper=config.VISUAL_CAP_AMOUNT)

    group_col = (
        map_feature_name(feature)
        if feature in ["Driver Age", "Vehicle Age", "Vehicle Power"]
        else feature
    )

    # Convert premiums to expected losses by multiplying by exposure
    # This lets us compare actual losses vs expected losses on same scale
    df_plot["ExpectedLoss_Model"] = df_plot["PurePremium_OB"] * df_plot[exposure_col]
    df_plot["ExpectedLoss_Final"] = df_plot["Final_Price"] * df_plot[exposure_col]

    agg = (
        df_plot.groupby(group_col, observed=False)
        .agg(
            {
                "TotalLoss_Visual": "sum",
                "ExpectedLoss_Model": "sum",
                "ExpectedLoss_Final": "sum",
                exposure_col: "sum",
            }
        )
        .reset_index()
    )

    # Convert back to rates (per unit exposure) for comparison
    agg["Actual_Rate"] = agg["TotalLoss_Visual"] / agg[exposure_col]
    agg["Model_Rate"] = agg["ExpectedLoss_Model"] / agg[exposure_col]
    agg["Final_Rate"] = agg["ExpectedLoss_Final"] / agg[exposure_col]

    agg = agg.sort_values(by=group_col).reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(config.PLOT_FIGSIZE_WIDTH, config.PLOT_FIGSIZE_HEIGHT))

    # Three-way comparison: actual experience vs model prediction vs final commercial price
    # Gap between actual and model shows where model under/over-estimates risk
    # Gap between model and final shows impact of actuarial loadings
    ax1.plot(
        agg[group_col].astype(str),
        agg["Actual_Rate"],
        color="red",
        marker="o",
        linewidth=2.5,
        label="Actual Average Loss",
        markersize=8,
    )
    ax1.plot(
        agg[group_col].astype(str),
        agg["Model_Rate"],
        color="blue",
        marker="s",
        linewidth=2.5,
        label="Model Predicted (PurePremium)",
        markersize=8,
    )
    ax1.plot(
        agg[group_col].astype(str),
        agg["Final_Rate"],
        color="green",
        marker="^",
        linewidth=2.5,
        label="Final Price (After Loadings)",
        markersize=8,
    )

    ax1.set_xlabel(f'{feature.replace("_Bin", "")} Bins', fontsize=11)
    ax1.set_ylabel("Average Cost per Unit Exposure (€)", fontsize=11)
    ax1.set_title(
        f'One-Way Analysis: {feature.replace("_Bin", "")}', fontsize=13, fontweight="bold"
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_xticks(range(len(agg)))
    ax1.set_xticklabels(agg[group_col].astype(str), rotation=45, ha="right")

    # Show exposure volume on secondary axis - helps assess statistical significance
    # Bins with low exposure have less reliable actual rates
    ax2 = ax1.twinx()
    ax2.bar(
        agg[group_col].astype(str),
        agg[exposure_col],
        alpha=0.2,
        color="gray",
        label="Exposure Volume",
        width=0.6,
    )
    ax2.set_ylabel("Exposure Volume", fontsize=11, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=9)

    plt.tight_layout()

    if return_fig:
        return fig

    if filename:
        if not os.path.isabs(filename):
            plots_dir = get_plots_dir()
            filename = str(plots_dir / Path(filename).name)
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved one-way analysis to {filename}")
        plt.close()
        return agg
    else:
        plt.close()

    return agg


def plot_dislocation_histogram(df, exposure_col="Exposure", return_fig=True):
    """
    Creates a histogram showing the percentage change between Technical and Commercial pricing.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing PurePremium_OB and Final_Price columns
    exposure_col : str
        Column name for exposure (default: 'Exposure')
    return_fig : bool, default True
        If True, returns matplotlib figure object

    Returns:
    --------
    matplotlib.figure.Figure object
    """
    df_plot = df.copy()

    df_plot["PctChange"] = (
        (df_plot["Final_Price"] - df_plot["PurePremium_OB"]) / df_plot["PurePremium_OB"] * 100
    )

    fig, ax = plt.subplots(figsize=(config.PLOT_FIGSIZE_WIDTH, config.PLOT_FIGSIZE_HEIGHT))

    # Separate increases/decreases to show impact of loadings visually
    # Red = price increases (loadings applied), Green = discounts
    increases = df_plot[df_plot["PctChange"] > 0]
    decreases = df_plot[df_plot["PctChange"] < 0]

    bins = np.linspace(df_plot["PctChange"].min(), df_plot["PctChange"].max(), 50)

    if len(increases) > 0:
        ax.hist(
            increases["PctChange"],
            bins=bins,
            alpha=0.7,
            color="red",
            label=f"Price Increases (n={len(increases)})",
            weights=increases[exposure_col],
        )
    if len(decreases) > 0:
        ax.hist(
            decreases["PctChange"],
            bins=bins,
            alpha=0.7,
            color="green",
            label=f"Price Decreases (n={len(decreases)})",
            weights=decreases[exposure_col],
        )

    ax.axvline(x=0, color="black", linestyle="--", linewidth=2, label="No Change (0%)")

    ax.set_xlabel("Percentage Change: Final Price vs Technical Price (%)", fontsize=11)
    ax.set_ylabel(f"Weighted Frequency (by {exposure_col})", fontsize=11)
    ax.set_title(
        "Dislocation Analysis: Commercial vs Technical Pricing", fontsize=13, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    plots_dir = get_plots_dir()
    filename = plots_dir / "dislocation_histogram.png"
    fig.savefig(str(filename), dpi=300, bbox_inches="tight")
    print(f"Saved dislocation histogram to {filename}")

    if return_fig:
        return fig
    else:
        plt.close()


def plot_lorenz_curve(
    df, pred_col="Final_Price", actual_col="TotalLoss", exposure_col="Exposure", return_fig=True
):
    """
    Creates a Lorenz Curve showing the cumulative distribution of losses vs exposure.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing predictions, actuals, and exposure
    pred_col : str
        Column name for predicted values (default: 'Final_Price')
    actual_col : str
        Column name for actual losses (default: 'TotalLoss')
    exposure_col : str
        Column name for exposure (default: 'Exposure')
    return_fig : bool, default True
        If True, returns matplotlib figure object

    Returns:
    --------
    matplotlib.figure.Figure object
    """
    data = df.copy()

    # Sort by predicted risk (lowest to highest) - this is key for Lorenz curve
    # If model is good, low-risk policies should have low actual losses
    data["pred_rate"] = data[pred_col] / data[exposure_col]
    data = data.sort_values(by="pred_rate", ascending=True).reset_index(drop=True)

    cum_exposure = data[exposure_col].cumsum() / data[exposure_col].sum()
    cum_losses = data[actual_col].cumsum() / data[actual_col].sum()

    # Add origin point for proper curve visualization
    cum_exposure = np.concatenate(([0], cum_exposure.values))
    cum_losses = np.concatenate(([0], cum_losses.values))

    # Gini = 1 - 2*A where A is area under Lorenz curve
    # Higher Gini = better risk differentiation
    # Version-agnostic: NumPy 2.0 uses trapezoid, 1.x uses trapz
    if hasattr(np, "trapezoid"):
        area_under_curve = np.trapezoid(cum_losses, cum_exposure)
    else:
        area_under_curve = np.trapz(cum_losses, cum_exposure)
    gini = 1 - (2 * area_under_curve)

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(
        cum_exposure,
        cum_losses,
        linewidth=2.5,
        color="blue",
        label=f"Lorenz Curve (Gini: {gini:.4f})",
    )

    # Diagonal line represents perfect equality (no risk differentiation)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Equality (Gini: 0.0000)")

    ax.fill_between(cum_exposure, cum_losses, cum_exposure, alpha=0.3, color="blue")

    ax.set_xlabel("Cumulative Proportion of Exposure", fontsize=9)
    ax.set_ylabel("Cumulative Proportion of Losses", fontsize=9)
    ax.set_title("Lorenz Curve: Risk Differentiation", fontsize=11, fontweight="bold")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    plots_dir = get_plots_dir()
    filename = plots_dir / "lorenz_curve.png"
    fig.savefig(str(filename), dpi=300, bbox_inches="tight")
    print(f"Saved Lorenz curve to {filename}")

    if return_fig:
        return fig
    else:
        plt.close()


def apply_loadings(row, custom_loadings=None):
    """
    Applies actuarial loadings (and discounts) to align
    technical price with observed one-way risk.

    Uses the ACTUARIAL_LOADINGS configuration dictionary to dynamically
    apply multipliers based on feature bin values.

    Parameters:
    -----------
    row : pd.Series
        Row of dataframe containing pricing data
    custom_loadings : dict, optional
        Custom loadings dictionary to override config.ACTUARIAL_LOADINGS
        Format: {'DriverAge_Bin': {'18-21': 2.4}, ...}
    """
    price = row["PurePremium_Calibrated"]

    loadings_to_apply = (
        custom_loadings if custom_loadings is not None else config.ACTUARIAL_LOADINGS
    )

    # Apply loadings multiplicatively - if multiple features match, they compound
    # e.g., young driver (2.4x) + high power (1.3x) = 3.12x total
    # NOTE: This assumes independence. In practice, might want interaction terms instead
    for feature_col, loading_rules in loadings_to_apply.items():
        bin_value = row.get(feature_col)

        if bin_value in loading_rules:
            multiplier = loading_rules[bin_value]
            price = price * multiplier

    return price


def calculate_gini(df, actual_col, pred_col, exposure_col="Exposure"):
    """
    Calculates the Gini coefficient for an insurance model.
    A higher Gini indicates better ability to differentiate between low and high risks.
    """
    data = df.copy()

    # Sort by predicted risk (lowest to highest) - this ordering is critical
    # Good model: low predicted risk → low actual losses
    data["pred_rate"] = data[pred_col] / data[exposure_col]
    data = data.sort_values(by="pred_rate", ascending=True)

    # Normalize cumulative distributions to [0, 1] for Lorenz curve
    cum_exposure = data[exposure_col].cumsum() / data[exposure_col].sum()
    cum_actual = data[actual_col].cumsum() / data[actual_col].sum()

    # Add origin for proper integration
    cum_exposure = np.concatenate(([0], cum_exposure.values))
    cum_actual = np.concatenate(([0], cum_actual.values))

    # Gini = 1 - 2*A where A is area under Lorenz curve
    # Perfect model (all losses in highest risk policies): Gini → 1
    # No differentiation (losses proportional to exposure): Gini → 0
    # Version-agnostic: NumPy 2.0 uses trapezoid, 1.x uses trapz
    if hasattr(np, "trapezoid"):
        area_under_curve = np.trapezoid(cum_actual, cum_exposure)
    else:
        area_under_curve = np.trapz(cum_actual, cum_exposure)
    gini = 1 - (2 * area_under_curve)

    return gini


def plot_lift_curve(
    df,
    pred_col,
    actual_col="TotalLoss",
    exposure_col="Exposure",
    title="Lift Chart",
    filename=None,
    view_type="Commercial",
    return_fig=False,
):
    """
    Generates a professional lift chart comparing predicted vs actual rates across risk deciles.

    Creates bins of equal exposure (not equal quantiles) to ensure fair comparison.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing predictions, actuals, and exposure
    pred_col : str
        Column name for predicted values (e.g., 'PurePremium_OB' or 'Final_Price')
    actual_col : str
        Column name for actual losses (default: 'TotalLoss')
    exposure_col : str
        Column name for exposure (default: 'Exposure')
    title : str
        Chart title
    filename : str
        Filename to save the chart (if None, displays instead)
    view_type : str
        'Technical' or 'Commercial' - affects labeling
    """
    data = df.copy()

    # Cap actuals to prevent single massive claim from making chart unreadable
    data["ActualLoss_Visual"] = data[actual_col].clip(upper=config.VISUAL_CAP_AMOUNT)

    data["Predicted_Rate"] = data[pred_col]
    data = data.sort_values(by="Predicted_Rate", ascending=True).reset_index(drop=True)

    # Create equal-exposure bins (not equal quantiles) - this is important!
    # Equal quantiles would give different exposure per bin, making rates harder to compare
    # Equal exposure ensures each decile represents same volume, so rate differences are meaningful
    total_exposure = data[exposure_col].sum()
    target_exposure_per_bin = total_exposure / config.RISK_BUCKET_COUNT

    data["CumulativeExposure"] = data[exposure_col].cumsum()
    data["RiskBucket"] = (data["CumulativeExposure"] / target_exposure_per_bin).astype(int)
    data["RiskBucket"] = data["RiskBucket"].clip(upper=config.RISK_BUCKET_COUNT - 1)

    lift_agg = (
        data.groupby("RiskBucket")
        .agg(
            {
                exposure_col: "sum",
                "ActualLoss_Visual": "sum",
            }
        )
        .reset_index()
    )

    # Exposure-weighted average predicted rate per bucket
    # Policies with more exposure should influence the bucket average more
    predicted_weighted = (
        data.groupby("RiskBucket")
        .apply(
            lambda x: (x[pred_col] * x[exposure_col]).sum() / x[exposure_col].sum(),
            include_groups=False,
        )
        .reset_index(name="Predicted_Rate")
    )
    lift_agg = lift_agg.merge(predicted_weighted[["RiskBucket", "Predicted_Rate"]], on="RiskBucket")

    lift_agg["Actual_Rate"] = lift_agg["ActualLoss_Visual"] / lift_agg[exposure_col]

    lift_agg["RiskBucket"] = lift_agg["RiskBucket"] + 1

    fig, ax1 = plt.subplots(figsize=(config.PLOT_FIGSIZE_WIDTH, config.PLOT_FIGSIZE_HEIGHT))

    # Good model: predicted and actual rates should both increase across deciles
    # Gap between lines shows calibration quality - should be relatively constant
    ax1.plot(
        lift_agg["RiskBucket"],
        lift_agg["Predicted_Rate"],
        color="green",
        marker="o",
        linewidth=2.5,
        label="Predicted",
        markersize=8,
    )
    ax1.plot(
        lift_agg["RiskBucket"],
        lift_agg["Actual_Rate"],
        color="#1f77b4",
        marker="s",
        linewidth=2.5,
        label="Actual",
        markersize=8,
    )
    ax1.set_xlabel("Risk Decile (1 = Lowest Risk, 10 = Highest Risk)", fontsize=11)
    ax1.set_ylabel("Average Cost per Unit Exposure (€)", fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_xticks(range(1, config.RISK_BUCKET_COUNT + 1))

    # Show exposure volume to assess statistical significance of each decile
    ax2 = ax1.twinx()
    ax2.bar(
        lift_agg["RiskBucket"],
        lift_agg[exposure_col],
        alpha=0.2,
        color="gray",
        label="Exposure Volume",
        width=0.6,
    )
    ax2.set_ylabel("Exposure Volume", fontsize=11, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=9)

    plt.tight_layout()

    if return_fig:
        return fig

    if filename:
        if not os.path.isabs(filename):
            plots_dir = get_plots_dir()
            filename = str(plots_dir / Path(filename).name)
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved lift chart to {filename}")
        plt.close()
    else:
        plt.close()


def run_pricing_pipeline(custom_loadings=None, return_train_data=False):
    """
    Runs the complete pricing pipeline and returns results.

    Parameters:
    -----------
    custom_loadings : dict, optional
        Custom loadings dictionary to override config.ACTUARIAL_LOADINGS
        Format: {'DriverAge_Bin': {'18-21': 2.4}, ...}
    return_train_data : bool, default False
        If True, also returns training data and models

    Returns:
    --------
    dict containing:
        - test_results: pd.DataFrame with pricing results
        - gini_technical: float
        - gini_commercial: float
        - avg_premium: float
        - models: dict with freq_m, sev_m, calibrator (if return_train_data=True)
        - train_data: pd.DataFrame (if return_train_data=True)
    """
    df_f, df_s = data_preprocessing.fetch_raw_data()
    df = data_preprocessing.preprocess_data(df_f, df_s)
    df = data_preprocessing.bin_features(df)

    # Standard 80/20 split with fixed random_state for reproducibility
    train, test = train_test_split(df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

    freq_m, feat_f = frequency_model.train_frequency_model(train)
    sev_m, feat_s = severity_model.train_severity_model(train)

    train["PredFreq"] = freq_m.predict(train[feat_f])
    train["PredSev"] = sev_m.predict(train[feat_s])
    train["PurePremium_Raw"] = train["PredFreq"] * train["PredSev"]

    # Off-balance adjustment: scale predictions so total predicted losses match actual
    # This is standard actuarial practice - ensures model is "balanced" on training data
    # Factor typically close to 1.0 if model is well-calibrated
    actual_total_train = train["TotalLoss"].sum()
    predicted_total_train = (train["PurePremium_Raw"] * train["Exposure"]).sum()
    off_balance_factor = actual_total_train / predicted_total_train
    train["PurePremium_OB"] = train["PurePremium_Raw"] * off_balance_factor

    # Isotonic calibration: non-parametric adjustment to improve calibration
    # Ensures monotonic relationship between predicted and actual rates
    calibrator = train_calibrator(
        predictions=train["PurePremium_OB"],
        actuals=train["TotalLoss"],
        exposure=train["Exposure"],
        cap_threshold=config.CALIBRATION_CAP_THRESHOLD,
    )

    test_results = test.copy()
    test_results["PredFreq"] = freq_m.predict(test[feat_f])
    test_results["PredSev"] = sev_m.predict(test[feat_s])

    test_results["PurePremium_Raw"] = test_results["PredFreq"] * test_results["PredSev"]
    test_results["PurePremium_OB"] = test_results["PurePremium_Raw"] * off_balance_factor
    test_results["PurePremium_Calibrated"] = calibrator.transform(test_results["PurePremium_OB"])

    if custom_loadings is not None or config.ENABLE_ACTUARIAL_LOADINGS:
        test_results["Final_Price"] = test_results.apply(
            lambda row: apply_loadings(row, custom_loadings), axis=1
        )
    else:
        test_results["Final_Price"] = test_results["PurePremium_Calibrated"]

    gini_technical = calculate_gini(
        test_results, actual_col="TotalLoss", pred_col="PurePremium_OB", exposure_col="Exposure"
    )

    gini_commercial = calculate_gini(
        test_results, actual_col="TotalLoss", pred_col="Final_Price", exposure_col="Exposure"
    )

    # Exposure-weighted average premium
    avg_premium = (test_results["Final_Price"] * test_results["Exposure"]).sum() / test_results[
        "Exposure"
    ].sum()

    result = {
        "test_results": test_results,
        "gini_technical": gini_technical,
        "gini_commercial": gini_commercial,
        "avg_premium": avg_premium,
        "off_balance_factor": off_balance_factor,
    }

    if return_train_data:
        result["models"] = {
            "freq_m": freq_m,
            "sev_m": sev_m,
            "calibrator": calibrator,
            "feat_f": feat_f,
            "feat_s": feat_s,
        }
        result["train_data"] = train

    # TODO: Consider adding model persistence here - currently models are retrained every run
    # Could save to pickle/joblib for faster iteration during development

    return result


def main():
    print("Loading data...")
    df_f, df_s = data_preprocessing.fetch_raw_data()
    df = data_preprocessing.preprocess_data(df_f, df_s)
    print("Binning features...")
    df = data_preprocessing.bin_features(df)

    train, test = train_test_split(df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

    print("Training Frequency Model...")
    freq_m, feat_f = frequency_model.train_frequency_model(train)

    print("Training Severity Model...")
    sev_m, feat_s = severity_model.train_severity_model(train)

    print("Generating Training Predictions for Calibration...")
    train["PredFreq"] = freq_m.predict(train[feat_f])
    train["PredSev"] = sev_m.predict(train[feat_s])
    train["PurePremium_Raw"] = train["PredFreq"] * train["PredSev"]

    actual_total_train = train["TotalLoss"].sum()
    predicted_total_train = (train["PurePremium_Raw"] * train["Exposure"]).sum()
    off_balance_factor = actual_total_train / predicted_total_train

    print(f"Off-Balance Factor: {off_balance_factor:.4f}")
    train["PurePremium_OB"] = train["PurePremium_Raw"] * off_balance_factor

    print("Fitting Isotonic Calibration Layer (With Capping)...")
    calibrator = train_calibrator(
        predictions=train["PurePremium_OB"],
        actuals=train["TotalLoss"],
        exposure=train["Exposure"],
        cap_threshold=config.CALIBRATION_CAP_THRESHOLD,
    )

    print("Applying to Test Set...")
    test_results = test.copy()
    test_results["PredFreq"] = freq_m.predict(test[feat_f])
    test_results["PredSev"] = sev_m.predict(test[feat_s])

    test_results["PurePremium_Raw"] = test_results["PredFreq"] * test_results["PredSev"]
    test_results["PurePremium_OB"] = test_results["PurePremium_Raw"] * off_balance_factor
    test_results["PurePremium_Calibrated"] = calibrator.transform(test_results["PurePremium_OB"])

    if config.ENABLE_ACTUARIAL_LOADINGS:
        print("Applying Actuarial Loadings for High Risk Segments...")
        test_results["Final_Price"] = test_results.apply(
            lambda row: apply_loadings(row, None), axis=1
        )

        avg_before = test_results["PurePremium_Calibrated"].mean()
        avg_after = test_results["Final_Price"].mean()
        total_before = (test_results["PurePremium_Calibrated"] * test_results["Exposure"]).sum()
        total_after = (test_results["Final_Price"] * test_results["Exposure"]).sum()

        print("\n--- Actuarial Loadings Summary ---")
        print(f"Average Price (Before Loadings): €{avg_before:.2f}")
        print(f"Average Price (After Loadings):  €{avg_after:.2f}")
        print(f"Total Premium (Before Loadings): €{total_before:,.2f}")
        print(f"Total Premium (After Loadings):  €{total_after:,.2f}")
        print(f"Overall Impact: {((avg_after / avg_before) - 1) * 100:+.2f}%")
        print("-" * 40 + "\n")
    else:
        test_results["Final_Price"] = test_results["PurePremium_Calibrated"]

    plots_dir = get_plots_dir()

    print("\n" + "=" * 50)
    print("MODEL VALIDATION: Gini Coefficients")
    print("=" * 50)

    # Technical Gini measures pure model performance (before loadings)
    gini_technical = calculate_gini(
        test_results, actual_col="TotalLoss", pred_col="PurePremium_OB", exposure_col="Exposure"
    )

    # Commercial Gini measures final pricing performance (after loadings)
    # If loadings are effective, this should be >= technical Gini
    gini_commercial = calculate_gini(
        test_results, actual_col="TotalLoss", pred_col="Final_Price", exposure_col="Exposure"
    )

    print(f"Technical Gini (PurePremium):  {gini_technical:.4f}")
    print(f"Commercial Gini (Final Price): {gini_commercial:.4f}")
    print(f"Gini Improvement:              {gini_commercial - gini_technical:+.4f}")
    print("=" * 50 + "\n")

    print("Generating Lift Charts...")

    plot_lift_curve(
        df=test_results,
        pred_col="PurePremium_OB",
        actual_col="TotalLoss",
        title="Lift Chart: Technical View (Pure Premium vs Actual)",
        filename=str(plots_dir / "lift_chart_technical.png"),
        view_type="Technical",
    )
    plt.close()

    plot_lift_curve(
        df=test_results,
        pred_col="Final_Price",
        actual_col="TotalLoss",
        title="Lift Chart: Commercial View (Final Price vs Actual)",
        filename=str(plots_dir / "lift_chart_commercial.png"),
        view_type="Commercial",
    )
    plt.close()

    print("Lift charts saved successfully.\n")

    print("\n" + "=" * 50)
    print("ONE-WAY ANALYSIS: Risk Gap Visualization")
    print("=" * 50)

    print("\nGenerating One-Way Analysis for DriverAge...")
    driver_age_agg = plot_one_way(
        df=test_results, feature="DriverAge_Bin", filename=str(plots_dir / "oneway_DriverAge.png")
    )
    plt.close()

    print("Generating One-Way Analysis for VehicleAge...")
    plot_one_way(
        df=test_results, feature="VehAge_Bin", filename=str(plots_dir / "oneway_VehicleAge.png")
    )
    plt.close()

    # Verification for high-risk bin - shows loadings are working as intended
    print("\n" + "-" * 50)
    print("VERIFICATION: DriverAge 18-21 Bin")
    print("-" * 50)

    driver_18_21 = driver_age_agg[driver_age_agg["DriverAge_Bin"] == "18-21"].iloc[0]

    print("\nBin: 18-21")
    print(f"Actual Average Loss:     €{driver_18_21['Actual_Rate']:.2f}")
    print(f"Model Predicted Rate:    €{driver_18_21['Model_Rate']:.2f}")
    print(f"Final Price Rate:        €{driver_18_21['Final_Rate']:.2f}")
    risk_gap = driver_18_21['Actual_Rate'] - driver_18_21['Model_Rate']
    loading_impact = driver_18_21['Final_Rate'] - driver_18_21['Model_Rate']
    loading_mult = driver_18_21['Final_Rate'] / driver_18_21['Model_Rate']
    print(f"\nRisk Gap (Actual - Model):     €{risk_gap:.2f}")
    print(f"Loading Impact (Final - Model): €{loading_impact:.2f}")
    print(f"Loading Multiplier:             {loading_mult:.2f}x")
    print(f"Exposure Volume:                {driver_18_21['Exposure']:,.2f}")
    print("-" * 50 + "\n")

    print("One-way analysis complete. Charts saved to plots/ directory.")

    print("\n" + "=" * 50)
    print("GENERATING ADDITIONAL VISUALIZATIONS")
    print("=" * 50)

    print("\nGenerating Dislocation Histogram...")
    plot_dislocation_histogram(df=test_results, return_fig=False)
    plt.close()

    print("Generating Lorenz Curve...")
    plot_lorenz_curve(df=test_results, pred_col="Final_Price", return_fig=False)
    plt.close()

    print("\nAll visualizations saved to plots/ directory.")


if __name__ == "__main__":
    main()
