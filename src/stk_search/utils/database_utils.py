""" script to load data from database or files"""


import numpy as np
import pandas as pd
import pymongo

# read data in database
def load_data_database( df_precursor_loc = "Data/calculation_data_precursor_310823_clean.pkl",num_fragm=6):
    if num_fragm == 6:
        collection_name = "BO_exp1"
    else:
        collection_name = f"BO_{num_fragm}"
    def load_data():
        client = pymongo.MongoClient("mongodb://ch-atarzia.ch.ic.ac.uk/")
        database = client["stk_mohammed_BO"]
        collection = database[f"{collection_name}_IPEA"]
        df_IPEA = pd.DataFrame(list(collection.find()))
        collection = database[f"{collection_name}_Stda"]
        df_STDA = pd.DataFrame(list(collection.find()))
        collection = database["constructed_molecules"]
        df_CM = pd.DataFrame(list(collection.find()))
        df_total = df_CM.merge(df_STDA, on="InChIKey", how="outer")
        df_total = df_total.merge(df_IPEA, on="InChIKey", how="outer")
        df_total.dropna(subset=["Excited state energy (eV)"], inplace=True)

        df_total["ES1"] = df_total["Excited state energy (eV)"].apply(
            lambda x: x[0]
        )
        df_total["fosc1"] = df_total[
            "Excited state oscillator strength"
        ].apply(lambda x: x[0])
        return df_total

    df_total_new = load_data()
    df_total_new.dropna(subset=["fosc1", "BB"], inplace=True)
    df_total_new = df_total_new[df_total_new["fosc1"] > 0]
    df_total_new = df_total_new[df_total_new["fosc1"] < 11]
    df_total_new["target"] = (
        -np.abs(df_total_new["ionisation potential (eV)"].values - 5.5)
        - 1 * np.abs(df_total_new["fosc1"].values - 10)
        - 0.5 * np.abs(df_total_new["ES1"].values - 3)
    )

    def prepare_df_for_plot(df_total_new=df_total_new):
        df_test = df_total_new
        for id, x in df_test.iterrows():
            # print(x)
            if len(x["BB"]) != num_fragm:
                df_test.drop(id, inplace=True)
        for i in range(num_fragm):
            df_test[f"InChIKey_{i}"] = df_test["BB"].apply(
                lambda x: x[i]["InChIKey"]
            )
        df_precursors = pd.read_pickle(
           df_precursor_loc
        )
        features_frag = df_precursors.columns[1:7].append(
            df_precursors.columns[17:23]
        )
        features_frag = features_frag.append(df_precursors.columns[0:1])
        for i in range(num_fragm):
            df_test = df_test.merge(
                df_precursors[features_frag].add_suffix(f"_{i}"),
                on=f"InChIKey_{i}",
                how="left",
            )
        return df_test, df_precursors

    df_total, df_precursors = prepare_df_for_plot(df_total_new)
    return df_total, df_precursors


# or load data from files
def load_data_from_file(
    df_path="",
    df_precursors_path="Data/calculation_data_precursor_310823_clean.pkl",
    features_frag=None,
    add_feature_frag=True,
    num_fragm=6,
):
    def prepare_df_for_plot(
        df_total_new: pd.DataFrame = [], features_frag=features_frag
    ):
        df_test = df_total_new
        for id, x in df_test.iterrows():
            # print(x)
            if len(eval(x["BB"])) != num_fragm:
                df_test.drop(id, inplace=True)
        for i in range(num_fragm):
            df_test[f"InChIKey_{i}"] = df_test["BB"].apply(
                lambda x: eval(x)[i]["InChIKey"]
            )
        df_precursors = pd.read_pickle(df_precursors_path)
        if features_frag is None:
            features_frag = df_precursors.columns[1:7].append(
                df_precursors.columns[17:23]
            )
            features_frag = features_frag.append(df_precursors.columns[0:1])
        else:
            features_frag = features_frag.append(df_precursors.columns[0:1])

        for i in range(num_fragm):
            df_test = df_test.merge(
                df_precursors[features_frag].add_suffix(f"_{i}"),
                on=f"InChIKey_{i}",
                how="left",
            )
        return df_test, df_precursors
    if df_path =="":
        df_precursors = pd.read_pickle(df_precursors_path)
        return  None, df_precursors
    else:
        df_total = pd.read_csv(df_path)
        df_total.dropna(subset=["fosc1", "BB"], inplace=True)
        df_total = df_total[df_total["fosc1"] > 0]
        df_total = df_total[df_total["fosc1"] < 11]
        df_total["target"] = (
            -np.abs(df_total["ionisation potential (eV)"].values - 5.5)
            - 1 * np.abs(df_total["fosc1"].values - 10)
            - 0.5 * np.abs(df_total["ES1"].values - 3)
        )
        if add_feature_frag:
            df_total, df_precursors = prepare_df_for_plot(
                df_total, features_frag=features_frag
            )
        else:
            df_precursors = pd.read_pickle(df_precursors_path)
        return df_total, df_precursors

def load_precursors_df(
    df_precursors_path="Data/calculation_data_precursor_310823_clean.pkl"):

    df_precursors = pd.read_pickle(df_precursors_path)

    return  df_precursors


def save_data(df_total,stk_path='/rds/general/user/ma11115/home/STK_Search/STK_search'):
    import datetime

    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d")
    df_total.to_csv(stk_path+f"/data/output/Full_dataset/df_total_{now}.csv")
    return f"df_total_{now}"