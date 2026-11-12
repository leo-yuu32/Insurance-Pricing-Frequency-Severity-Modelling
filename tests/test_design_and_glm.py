import numpy as np
import pandas as pd
import pytest

from design import BANDS, CLAIM_FREE_LADDER, Design, _band_labels, add_model_columns
from glm import fit_gamma, fit_poisson, predict


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 20000
    df = pd.DataFrame({
        "Colour": rng.choice(["red", "blue", "green"], n, p=[0.2, 0.5, 0.3]),
        "Size": rng.choice(["s", "l"], n, p=[0.7, 0.3]),
        "x": rng.normal(0.0, 1.0, n),
        "Exposure": rng.uniform(0.1, 1.0, n),
    })
    rate = np.exp(-2.0 + 0.5 * (df["Colour"] == "red") - 0.3 * (df["Colour"] == "green") + 0.4 * (df["Size"] == "l") + 0.2 * df["x"])
    df["ClaimNb"] = rng.poisson(rate * df["Exposure"])
    return df


def test_band_labels():
    assert _band_labels([17, 19, 21, 200]) == ["18-19", "20-21", "22+"]
    assert _band_labels([49, 50, 54, 400]) == ["50", "51-54", "55+"]


def test_every_policy_gets_a_band():
    df = pd.DataFrame({"DrivAge": [18.0, 19.0, 20.0, 99.0], "VehAge": [0.0, 1.0, 13.0, 100.0], "VehPower": [4.0, 9.0, 12.0, 15.0],
                       "BonusMalus": [50.0, 51.0, 100.0, 230.0], "Density": [1.0, 10.0, 100.0, 27000.0]})
    out = add_model_columns(df)
    for name in BANDS:
        assert not (out[name] == "nan").any()
    assert out["DrivAgeBand"].tolist() == ["18-19", "18-19", "20-21", "76+"]
    assert out["BonusMalusBand"].tolist() == ["50", "51-54", "100", "120+"]
    assert out["BonusMalusCapped"].max() == 150.0


def test_claim_free_ladder():
    # 100 times 0.95 each claim-free year, truncated to two decimals, down to the floor of 50.
    ladder, value = [100], 1.00
    while value > 0.50:
        value = max(np.floor(value * 0.95 * 100 + 1e-9) / 100, 0.50)
        ladder.append(int(round(value * 100)))
    assert sorted(set(ladder)) == CLAIM_FREE_LADDER
    df = pd.DataFrame({"DrivAge": 40.0, "VehAge": 5.0, "VehPower": 6.0, "Density": 100.0,
                       "BonusMalus": [50.0, 62.0, 64.0, 63.0, 95.0, 100.0, 118.0]})
    assert add_model_columns(df)["BonusMalusOffLadder"].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_coarse_bands_are_merges_of_the_fine_bands():
    for coarse, fine in [("DrivAgeCoarse", "DrivAgeBand"), ("BonusMalusCoarse", "BonusMalusBand"),
                         ("VehAgeCoarse", "VehAgeBand"), ("VehPowerCoarse", "VehPowerBand")]:
        assert set(BANDS[coarse][1]) <= set(BANDS[fine][1]), coarse


def test_reference_level_is_the_highest_exposure_level(toy):
    design = Design(["Colour", "Size", "x"]).fit(toy)
    assert design.levels["Colour"][0] == "blue"
    assert design.levels["Size"][0] == "s"
    assert design.columns == ["Intercept", "Colour[green]", "Colour[red]", "Size[l]", "x"]


def test_transform_gives_the_same_columns_on_new_data(toy):
    design = Design(["Colour", "x"]).fit(toy)
    new = pd.DataFrame({"Colour": ["red", "blue", "purple"], "x": [1.0, 2.0, 3.0], "Exposure": [1.0, 1.0, 1.0]})
    X = design.transform(new)
    assert list(X.columns) == design.columns
    assert X.loc[0, "Colour[red]"] == 1.0
    assert X.loc[1, ["Colour[green]", "Colour[red]"]].sum() == 0.0   # reference level
    assert X.loc[2, ["Colour[green]", "Colour[red]"]].sum() == 0.0   # unseen level is priced as the reference


def test_interaction_columns_are_products(toy):
    design = Design(["Colour", "Size", "x"], interactions=[("Colour", "Size"), ("Size", "x")]).fit(toy)
    X = design.transform(toy)
    assert "Colour[red]:Size[l]" in X.columns and "Size[l]:x" in X.columns
    np.testing.assert_array_equal(X["Colour[red]:Size[l]"], X["Colour[red]"] * X["Size[l]"])
    np.testing.assert_array_equal(X["Size[l]:x"], X["Size[l]"] * X["x"])
    assert np.linalg.matrix_rank(X.to_numpy()) == X.shape[1]


def test_poisson_glm_recovers_known_coefficients(toy):
    design = Design(["Colour", "Size", "x"]).fit(toy)
    result = fit_poisson(design.transform(toy), toy["ClaimNb"], toy["Exposure"])
    truth = np.array([-2.0, -0.3, 0.5, 0.4, 0.2])
    assert np.all(np.abs(result.params - truth) < 4 * result.bse)


def test_poisson_glm_reproduces_the_claim_total_of_every_level(toy):
    # Balance property of the Poisson log-link GLM with an intercept: on its training data,
    # predicted claims equal observed claims in total and for each level of each categorical factor.
    design = Design(["Colour", "Size", "x"]).fit(toy)
    result = fit_poisson(design.transform(toy), toy["ClaimNb"], toy["Exposure"])
    expected = predict(result, design.transform(toy)) * toy["Exposure"]
    assert expected.sum() == pytest.approx(toy["ClaimNb"].sum(), rel=1e-8)
    for factor in ["Colour", "Size"]:
        for level, rows in toy.groupby(factor).groups.items():
            assert expected[rows].sum() == pytest.approx(toy.loc[rows, "ClaimNb"].sum(), rel=1e-7), (factor, level)


def test_gamma_glm_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    n = 20000
    df = pd.DataFrame({"Colour": rng.choice(["red", "blue"], n, p=[0.3, 0.7]), "x": rng.normal(0.0, 1.0, n), "Exposure": 1.0})
    weights = rng.integers(1, 4, n)
    mean = np.exp(7.0 + 0.3 * (df["Colour"] == "red") - 0.1 * df["x"])
    shape = 2.0  # an average of w claims has shape w * shape
    y = rng.gamma(shape * weights, mean / (shape * weights))
    design = Design(["Colour", "x"]).fit(df)
    result = fit_gamma(design.transform(df), y, weights)
    assert np.all(np.abs(result.params - np.array([7.0, 0.3, -0.1])) < 4 * result.bse)
    assert result.pearson_chi2 / result.df_resid == pytest.approx(1.0 / shape, rel=0.1)
