"""One design-matrix builder for the GLMs (statsmodels and the NumPy solver fit the same matrix).

Banding and term choices are explained in docs/DECISIONS.md. All cut-points were chosen on
training data only.
"""

import numpy as np
import pandas as pd

# name of the banded column: (source column, right-closed bin edges)
BANDS = {
    "DrivAgeBand": ("DrivAge", [17, 19, 21, 23, 25, 30, 40, 50, 60, 75, 200]),
    "DrivAgeBand8": ("DrivAge", [17, 21, 25, 30, 40, 50, 60, 75, 200]),  # legacy bands, for comparison
    "VehAgeBand": ("VehAge", [-1, 1, 5, 12, 17, 200]),
    "VehPowerBand": ("VehPower", [3, 4, 5, 6, 7, 8, 9, 10, 11, 99]),
    "BonusMalusBand": ("BonusMalus", [49, 50, 54, 64, 79, 94, 99, 100, 119, 400]),
    # Coarse bands, used only inside interaction terms. Each is a merge of the fine bands above.
    "DrivAgeCoarse": ("DrivAge", [17, 25, 40, 200]),
    "BonusMalusCoarse": ("BonusMalus", [49, 50, 99, 400]),
    "VehAgeCoarse": ("VehAge", [-1, 1, 12, 200]),
    "VehPowerCoarse": ("VehPower", [3, 5, 8, 99]),
}
BONUS_MALUS_CAP = 150.0

# The baseline frequency GLM. Area is left out because it is a banding of Density.
FREQUENCY_TERMS = ["DrivAgeBand", "VehAgeBand", "VehPowerBand", "BonusMalusBand", "LogDensity", "VehBrand", "VehGas", "Region"]
# The severity GLM. Driver age is the only factor that improved cross-validated Gamma deviance
# over an intercept-only model (src/compare_glm_designs.py).
SEVERITY_TERMS = ["DrivAgeBand"]


def _band_labels(edges):
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo = int(lo) + 1
        if hi == edges[-1]:
            labels.append(f"{lo}+")
        elif lo == hi:
            labels.append(f"{lo}")
        else:
            labels.append(f"{lo}-{int(hi)}")
    return labels


def add_model_columns(df):
    """Add the banded and transformed columns the GLM terms refer to."""
    df = df.copy()
    for name, (source, edges) in BANDS.items():
        df[name] = pd.cut(df[source], bins=edges, labels=_band_labels(edges)).astype(str)
    df["LogDensity"] = np.log(df["Density"])
    df["BonusMalusCapped"] = df["BonusMalus"].clip(upper=BONUS_MALUS_CAP)
    return df


class Design:
    """Turns a list of terms into a numeric design matrix.

    A term is a column name: text columns become dummy variables with the highest-exposure
    training level as the reference, numeric columns enter as they are. `interactions` is a
    list of (term, term) pairs; each adds the products of the two terms' columns.
    """

    def __init__(self, terms, interactions=()):
        self.terms = list(terms)
        self.interactions = [tuple(pair) for pair in interactions]
        self.levels = {}
        self.columns = None

    def fit(self, train):
        needed = set(self.terms) | {t for pair in self.interactions for t in pair}
        for term in needed:
            if not pd.api.types.is_numeric_dtype(train[term]):
                exposure = train.groupby(term)["Exposure"].sum().sort_values(ascending=False)
                self.levels[term] = list(exposure.index)  # reference level first
        matrix = self._build(train)
        self.columns = [c for c in matrix.columns if c == "Intercept" or matrix[c].abs().sum() > 0]
        return self

    def _term_columns(self, df, term):
        if term in self.levels:
            return {f"{term}[{level}]": (df[term] == level).to_numpy(dtype=float) for level in self.levels[term][1:]}
        return {term: df[term].to_numpy(dtype=float)}

    def _build(self, df):
        cols = {"Intercept": np.ones(len(df))}
        for term in self.terms:
            cols.update(self._term_columns(df, term))
        for a, b in self.interactions:
            for name_a, col_a in self._term_columns(df, a).items():
                for name_b, col_b in self._term_columns(df, b).items():
                    cols[f"{name_a}:{name_b}"] = col_a * col_b
        return pd.DataFrame(cols, index=df.index)

    def transform(self, df):
        return self._build(df)[self.columns]

    def fit_transform(self, train):
        return self.fit(train).transform(train)
