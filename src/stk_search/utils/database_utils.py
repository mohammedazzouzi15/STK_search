"""script to load data from database or files."""

import numpy as np
import pandas as pd
import pymongo


# read data in database
def load_data_database(
    df_precursor_loc="Data/calculation_data_precursor_310823_clean.pkl",
    num_fragm=6,
    client_url="mongodb://localhost:27017/",
    database_name="stk_constructed",
    collection_name="BO_exp1",
):
    """Load data from database.

    Args:
    ----
        df_precursor_loc (str): path to the precursor data.
        num_fragm (int): number of fragments.
        client_url (str): url to the database.
        database_name (str): name of the database.
        collection_name (str): name of the collection.

    Returns:
    -------
        df_total (pd.DataFrame): dataframe with the data.
        df_precursors (pd.DataFrame): dataframe with the precursors.

    """

    def load_data() -> pd.DataFrame:
        client = pymongo.MongoClient(client_url)
        database = client[database_name]
        collection = database[f"{collection_name}_IPEA"]
        df_IPEA = pd.DataFrame(list(collection.find()))
        collection = database[f"{collection_name}_Stda"]
        df_STDA = pd.DataFrame(list(collection.find()))
        collection = database["constructed_molecules"]
        df_CM = pd.DataFrame(list(collection.find()))
        df_total = df_CM.merge(df_STDA, on="InChIKey", how="outer")

        df_total = df_total.merge(df_IPEA, on="InChIKey", how="outer")
        df_total = df_total.dropna(subset=["Excited state energy (eV)"])

        df_total["ES1"] = df_total["Excited state energy (eV)"].apply(
            lambda x: x[0]
        )
        df_total["fosc1"] = df_total[
            "Excited state oscillator strength"
        ].apply(lambda x: x[0])
        return df_total

    df_total_new = load_data()
    df_total_new = df_total_new.dropna(subset=["fosc1", "BB"])
    df_total_new = df_total_new[df_total_new["fosc1"] > 0]
    df_total_new = df_total_new[df_total_new["fosc1"] < 11]
    df_total_new["target"] = (
        -np.abs(df_total_new["ES1"] - 3)
        - np.abs(df_total_new["ionisation potential (eV)"] - 5.5)
        + np.log10(df_total_new["fosc1"])
    )

    def prepare_df_for_plot(df_total_new=df_total_new) -> tuple:
        df_test = df_total_new
        for _id, x in df_test.iterrows():
            if len(x["BB"]) != num_fragm:
                df_test = df_test.drop(_id)
        df_precursors = pd.read_pickle(df_precursor_loc)
        for i in range(num_fragm):
            df_test[f"InChIKey_{i}"] = df_test["BB"].apply(
                lambda x: str(x[i]["InChIKey"])
            )
            df_test = df_test[
                df_test[f"InChIKey_{i}"].isin(df_precursors["InChIKey"])
            ]
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
        df_total_new: pd.DataFrame = None, features_frag=features_frag
    ):
        if df_total_new is None:
            df_total_new = []
        df_test = df_total_new
        df_precursors = pd.read_pickle(df_precursors_path)
        if features_frag is None:
            # consider only columns that are np.number
            features_frag = df_precursors.select_dtypes(
                include=[np.number]
            ).columns
            features_frag = features_frag.append(pd.Index(["InChIKey"]))
        else:
            features_frag = features_frag.append(pd.Index(["InChIKey"]))

        for i in range(num_fragm):
            df_test = df_test.merge(
                df_precursors[features_frag].add_suffix(f"_{i}"),
                on=f"InChIKey_{i}",
                how="left",
                suffixes=("_old", ""),
            )
        return df_test, df_precursors

    if df_path == "":
        df_precursors = pd.read_pickle(df_precursors_path)
        return None, df_precursors
    df_total = pd.read_csv(df_path)
    if add_feature_frag:
        df_total, df_precursors = prepare_df_for_plot(
            df_total, features_frag=features_frag
        )
    else:
        df_precursors = pd.read_pickle(df_precursors_path)
    return df_total, df_precursors


def load_precursors_df(
    df_precursors_path="Data/calculation_data_precursor_310823_clean.pkl",
):
    return pd.read_pickle(df_precursors_path)


def save_data(
    df_total, stk_path="/rds/general/user/ma11115/home/STK_Search/STK_search"
):
    import datetime

    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d")
    df_total.to_csv(stk_path + f"/data/output/Full_dataset/df_total_{now}.csv")
    return f"df_total_{now}"
