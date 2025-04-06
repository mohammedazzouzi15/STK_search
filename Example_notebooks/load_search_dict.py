
import copy
import datetime
import glob
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from stk_search.utils import Search_results_plot


def join_name(x):
    return "_".join(x)


def get_results_length(x):
    try:

        return max(pd.read_pickle(x)["ids_acquired"])
    except Exception:
        return 0

def load_search_dict(search_exp_file):
    res = pickle.load(open(search_exp_file, "rb"))
    return res

if __name__ == "__main__":
    # Load the search dictionary
    search_exp_file = "data_example/data_benchmark/runs9/6_frag/BO_learned/20250405/results_f88c750fff1d4fd6b5dedaa3dee43ca8.pkl"
    search_dict = load_search_dict(search_exp_file)
    print(pd.read_pickle(search_exp_file)["ids_acquired"])
    print(search_dict.keys())
    for k, v in search_dict.items():
        print("shape of ", k, ":", len(v))