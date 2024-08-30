import pandas as pd
from stk_search import SearchedSpace

""" list of test to check the function in SearchSpace.py """


def test_init():
    df_precursors = pd.read_pickle("tests/data/df_precursor_test.pkl")
    SP = SearchedSpace.SearchedSpace(
        number_of_fragments=2,
        df=df_precursors,
        features_frag=df_precursors.columns[0:1],
        generation_type="conditional",
    )
    assert SP.number_of_fragments == 2
    assert SP.generation_type == "conditional"

    assert SP.get_space_size() == 100
