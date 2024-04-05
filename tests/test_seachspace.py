import pytest
import pandas as pd
from stk_search import Searched_space

""" list of test to check the function in Search_Space.py """


def test_init():
    df_precursors = pd.read_pickle("tests/data/df_precursor_test.pkl")
    SP = Searched_space.Searched_Space(
        number_of_fragments=2,
        df=df_precursors,
        features_frag=df_precursors.columns[0:1],
        generation_type="conditional",
    )
    assert SP.number_of_fragments == 2
    assert SP.generation_type == "conditional"

    assert SP.get_space_size() == 100
