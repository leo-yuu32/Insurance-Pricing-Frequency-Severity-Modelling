from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import config


def train_frequency_model(df_train):
    """
    Fits a Poisson GLM for Frequency (Claims per Year).
    Uses a Pipeline to handle categorical variables (One-Hot Encoding) automatically.
    """

    categorical_features = config.CATEGORICAL_FEATURES.copy()
    numerical_features = config.NUMERICAL_FEATURES.copy()

    # OneHotEncoder with drop='first' avoids dummy variable trap
    # handle_unknown='ignore' is crucial for production - new categories won't break predictions
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    # Poisson is the standard choice for count data in insurance
    # Small alpha prevents rare category combinations from getting extreme coefficients
    glm = PoissonRegressor(
        alpha=config.GLM_ALPHA, max_iter=config.MAX_ITER, solver=config.GLM_SOLVER
    )

    model_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", glm)])

    # Target is claim rate (claims per unit exposure), not raw count
    # This accounts for policies with different exposure durations
    y_train = df_train["ClaimNb"] / df_train["Exposure"]

    print("Fitting Frequency GLM...")

    # Weight by exposure so policies with longer duration have more influence
    # A policy with 1 year exposure should count more than one with 0.1 years
    model_pipeline.fit(
        df_train[categorical_features + numerical_features],
        y_train,
        regressor__sample_weight=df_train["Exposure"],
    )

    return model_pipeline, categorical_features + numerical_features
