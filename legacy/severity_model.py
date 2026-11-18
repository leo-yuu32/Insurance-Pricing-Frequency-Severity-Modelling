from sklearn.linear_model import GammaRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import config


def train_severity_model(df_train):
    """
    Fits a Gamma GLM for Severity (Cost per Claim).
    Uses a Pipeline to handle categorical variables (One-Hot Encoding) automatically.
    """

    # Only train on policies that actually had claims - can't model severity of zero claims
    df_nonzero = df_train[(df_train["ClaimNb"] > 0) & (df_train["AvgSeverity"] > 0)].copy()

    # Cap severity at 99.9th percentile to prevent massive outliers from
    # destabilising the Gamma model. This is standard actuarial practice -
    # we're modeling typical claims, not catastrophic events.
    # NOTE: Only training data is capped. Test predictions use raw severity
    # to evaluate true model performance
    cap = df_nonzero["AvgSeverity"].quantile(config.SEVERITY_CAP_PERCENTILE)
    df_nonzero["AvgSeverity"] = df_nonzero["AvgSeverity"].clip(upper=cap)
    print(f"Severity Capping Threshold ({config.SEVERITY_CAP_PERCENTILE*100}%): €{cap:,.2f}")

    categorical_features = config.CATEGORICAL_FEATURES.copy()
    numerical_features = config.NUMERICAL_FEATURES.copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    # Gamma distribution is standard for positive continuous costs (severity)
    # Regularization prevents rare feature combinations from getting extreme coefficients
    glm = GammaRegressor(alpha=config.GLM_ALPHA, max_iter=config.MAX_ITER, solver=config.GLM_SOLVER)

    model_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", glm)])

    print("Fitting Severity GLM...")

    # Weight by ClaimNb - a policy with 3 claims should influence the model
    # more than one with 1 claim. This ensures the model learns from policies
    # with more claim experience
    model_pipeline.fit(
        df_nonzero[categorical_features + numerical_features],
        df_nonzero["AvgSeverity"],
        regressor__sample_weight=df_nonzero["ClaimNb"],
    )

    # TODO: Consider adding interaction terms for high-risk combinations
    # (e.g., young driver + high power). Current model assumes independence
    # between features, which might not hold in practice

    return model_pipeline, categorical_features + numerical_features
