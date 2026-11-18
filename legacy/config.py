"""
Configuration file for Insurance Pricing Frequency-Severity Modelling.

This module centralizes all hardcoded parameters and constants used throughout
the project to improve maintainability and configurability.
"""

# ============================================================================
# DATA SOURCES
# ============================================================================

# OpenML Dataset IDs
OPENML_FREQUENCY_DATA_ID = 41214
OPENML_SEVERITY_DATA_ID = 41215

# ============================================================================
# DATA SPLITTING
# ============================================================================

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# GLM Hyperparameters (used for both Frequency and Severity models)
GLM_ALPHA = 0.01
GLM_SOLVER = "newton-cholesky"
MAX_ITER = 1000

# ============================================================================
# BUSINESS LOGIC & THRESHOLDS
# ============================================================================

# Severity capping
SEVERITY_CAP_PERCENTILE = 0.999

# Visual and calibration capping
VISUAL_CAP_AMOUNT = 15000
CALIBRATION_CAP_THRESHOLD = 15000

# Data filtering thresholds
MIN_EXPOSURE = 0.003
MAX_EXPOSURE = 1.0
MAX_CLAIM_NB = 10

# ============================================================================
# FEATURE BINNING CONFIGURATION
# ============================================================================

# Driver Age Binning
DRIVER_AGE_BINS = [0, 21, 25, 30, 40, 50, 60, 75, 120]
DRIVER_AGE_LABELS = ["18-21", "22-25", "26-30", "31-40", "41-50", "51-60", "61-75", "75+"]

# Vehicle Age Binning
VEHICLE_AGE_BINS = [-1, 1, 4, 10, 20, 100]
VEHICLE_AGE_LABELS = ["New (0-1)", "2-4", "5-10", "11-20", "20+"]

# Vehicle Power Binning
VEHICLE_POWER_THRESHOLD = 10  # Values >= 10 are grouped as '10+'

# Vehicle Brand Binning
TOP_BRANDS_COUNT = 5
EXCLUDED_BRAND = "B12"  # Brand to exclude even if in top N

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Plot dimensions
PLOT_FIGSIZE_WIDTH = 10
PLOT_FIGSIZE_HEIGHT = 6

# One-way analysis binning
ONE_WAY_NUMERIC_BINS = 10  # Number of bins for numeric features in one-way analysis
ONE_WAY_NUMERIC_THRESHOLD = 20  # Minimum unique values to trigger binning

# Lift chart deciles
RISK_BUCKET_COUNT = 10

# ============================================================================
# MODEL FEATURES
# ============================================================================

# Categorical features used in both Frequency and Severity models
CATEGORICAL_FEATURES = [
    "VehBrand_Bin",
    "VehGas",
    "Region",
    "Area",
    "VehPower_Bin",
    "VehAge_Bin",
    "DriverAge_Bin",
]

# Numerical features (currently empty, but can be extended)
NUMERICAL_FEATURES = []

# ============================================================================
# ACTUARIAL LOADINGS & DISCOUNTS
# ============================================================================

# Actuarial loadings/discounts applied to calibrated pure premium
# Keys represent the feature column name (binned), values are dictionaries
# mapping bin labels to multipliers
ACTUARIAL_LOADINGS = {
    "DriverAge_Bin": {
        "18-21": 2.40,  # 140% loading for young drivers
        "22-25": 1.40,  # 40% loading for young drivers
    },
    "VehAge_Bin": {
        "New (0-1)": 0.75,  # 25% discount for new cars
        "5-10": 1.12,  # 12% surcharge for mid-age vehicles
        "20+": 0.55,  # 45% discount for very old cars
    },
    "VehPower_Bin": {
        "9": 1.30,  # 30% surcharge for power level 9
        "4": 0.90,  # 10% discount for power level 4
    },
}

# Flag to enable/disable actuarial loadings
ENABLE_ACTUARIAL_LOADINGS = True
