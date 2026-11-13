import pandas as pd
import pytest

from check_bonus_malus_timing import ONE_SURCHARGE_ABOVE_LADDER, exposure_class, frequency_table, value_group
from design import CLAIM_FREE_LADDER


def test_one_surcharge_values():
    # 50 x 1.25 = 62.5 -> 62, 100 x 1.25 = 125. Values that land back on the ladder (64 -> 80) are left out.
    assert ONE_SURCHARGE_ABOVE_LADDER == [62, 63, 67, 71, 75, 106, 112, 118, 125]
    assert not set(ONE_SURCHARGE_ABOVE_LADDER) & set(CLAIM_FREE_LADDER)


def test_chain_from_50_is_the_claim_free_decay_of_62():
    value, chain = 62, []
    while value > 50:
        chain.append(value)
        value = max(int(value * 0.95 + 1e-9), 50)
    assert chain == [62, 58, 55, 52]
    for v in chain:
        assert value_group(v).startswith(str(v))


def test_value_groups():
    assert value_group(50.0) == "50"
    assert value_group(54.0) == "51, 54, 57 (claim-free values next to 50)"
    assert value_group(112.0) == "one surcharge above another claim-free value"
    assert value_group(80.0) == "other"


def test_exposure_class():
    assert list(exposure_class([1.0, 0.9995, 0.75, 0.5, 0.49, 0.003])) == [
        "full year", "full year", "half to one year", "half to one year", "under half a year", "under half a year"]


def test_frequency_table():
    df = pd.DataFrame({"g": ["a", "a", "a", "b"], "Exposure": [1.0, 0.5, 0.5, 0.25], "ClaimNb": [2, 0, 1, 1]})
    table = frequency_table(df, "g")
    assert table.loc["a", "policies"] == 3 and table.loc["a", "claims"] == 3
    assert table.loc["a", "frequency"] == pytest.approx(1.5)
    assert table.loc["a", "share_with_claim"] == pytest.approx(2 / 3)
    assert table.loc["b", "frequency"] == pytest.approx(4.0)
