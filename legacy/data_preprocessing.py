import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import config


def fetch_raw_data():
    """
    Loads the standard French Motor Third-Party Liability datasets.
    """
    print("Downloading dataset from OpenML...")
    freq = fetch_openml(data_id=config.OPENML_FREQUENCY_DATA_ID, as_frame=True, parser="auto").frame
    sev = fetch_openml(data_id=config.OPENML_SEVERITY_DATA_ID, as_frame=True, parser="auto").frame

    # OpenML sometimes returns IDs as strings, force to int for reliable merging
    freq["IDpol"] = freq["IDpol"].astype(int)
    sev["IDpol"] = sev["IDpol"].astype(int)

    return freq, sev


def preprocess_data(df_freq, df_sev):
    """
    Merges Frequency and Severity data and performs basic cleaning.
    """
    # Aggregate severity at policy level - one policy can have multiple claims
    sev_agg = df_sev.groupby("IDpol")["ClaimAmount"].sum().reset_index()
    sev_agg.rename(columns={"ClaimAmount": "TotalLoss"}, inplace=True)

    df = pd.merge(df_freq, sev_agg, on="IDpol", how="left")

    # OpenML dataset uses 'DrivAge' but our binning logic expects 'DriverAge'
    # This mismatch caused issues early on, so standardizing here
    df.rename(columns={"DrivAge": "DriverAge"}, inplace=True)

    # Policies with no claims have NaN TotalLoss after left merge
    df["TotalLoss"] = df["TotalLoss"].fillna(0)

    # AvgSeverity only meaningful when there are actual claims
    df["AvgSeverity"] = np.where(df["ClaimNb"] > 0, df["TotalLoss"] / df["ClaimNb"], 0)

    # Filter out policies with suspiciously low exposure (likely data errors)
    # Also cap exposure at 1.0 to handle any edge cases where it exceeds a full year
    df = df[df["Exposure"] > config.MIN_EXPOSURE].copy()
    df["Exposure"] = df["Exposure"].clip(upper=config.MAX_EXPOSURE)
    # Remove extreme outliers - policies with >10 claims are likely data quality issues
    df = df[df["ClaimNb"] < config.MAX_CLAIM_NB].copy()

    return df


def bin_features(df):
    """
    Applies standard actuarial binning.
    """
    df = df.copy()

    # Driver age bins follow standard actuarial practice: finer granularity for young drivers
    # where risk varies most dramatically
    df["DriverAge_Bin"] = pd.cut(
        df["DriverAge"],
        bins=config.DRIVER_AGE_BINS,
        labels=config.DRIVER_AGE_LABELS,
        include_lowest=True,
    )

    # Vehicle age bins - new cars get special treatment, very old cars grouped together
    df["VehAge_Bin"] = pd.cut(
        df["VehAge"], bins=config.VEHICLE_AGE_BINS, labels=config.VEHICLE_AGE_LABELS
    )

    # Vehicle power: individual values for lower power levels, group high power together
    # This prevents sparse categories while preserving granularity where it matters
    df["VehPower_Bin"] = df["VehPower"].apply(
        lambda x: f"{x}" if x < config.VEHICLE_POWER_THRESHOLD else "10+"
    )

    # Ensure categoricals are strings - prevents issues with mixed types or numeric codes
    categorical_cols = ["VehBrand", "VehGas", "Region", "Area"]
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    # Brand binning: keep top N brands as-is, everything else goes to 'Other'
    # B12 explicitly excluded even if in top N - had some data quality issues in early exploration
    top_brands = df["VehBrand"].value_counts().nlargest(config.TOP_BRANDS_COUNT).index

    df["VehBrand_Bin"] = df["VehBrand"].apply(
        lambda x: x if (x in top_brands and x != config.EXCLUDED_BRAND) else "Other"
    )

    # TODO: This binning logic is getting a bit scattered. Consider moving all bin definitions
    # to a single YAML config file for easier maintenance and version control.

    return df
