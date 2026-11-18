"""Settings shared by the pipeline. The legacy pipeline keeps its own settings in legacy/config.py."""

# OpenML dataset IDs: freMTPL2freq (policies) and freMTPL2sev (claims)
OPENML_FREQUENCY_DATA_ID = 41214
OPENML_SEVERITY_DATA_ID = 41215

# The one seed used by every random step (split, tuning search, bootstrap, SHAP samples).
SEED = 42
