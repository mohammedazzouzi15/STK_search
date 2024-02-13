import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import pymongo

# read data in database
def load_data_database( df_precursor_loc = "Data/calculation_data_precursor_310823_clean.pkl"):
    def load_data():
        client = pymongo.MongoClient("mongodb://129.31.66.201/")
        database = client["stk_mohammed_BO"]
        collection = database["BO_exp1_IPEA"]
        df_IPEA = pd.DataFrame(list(collection.find()))
        collection = database["BO_exp1_Stda"]
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
            if len(x["BB"]) < 6:
                df_test.drop(id, inplace=True)
        for i in range(6):
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
        for i in range(6):
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
    df_path="Data/df_total_new.csv2023_08_20",
    df_precursors_path="Data/calculation_data_precursor_310823_clean.pkl",
    features_frag=None,
):
    def prepare_df_for_plot(
        df_total_new: pd.DataFrame = [], features_frag=features_frag
    ):
        df_test = df_total_new
        for id, x in df_test.iterrows():
            # print(x)
            if len(eval(x["BB"])) < 6:
                df_test.drop(id, inplace=True)
        for i in range(6):
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

        for i in range(6):
            df_test = df_test.merge(
                df_precursors[features_frag].add_suffix(f"_{i}"),
                on=f"InChIKey_{i}",
                how="left",
            )
        return df_test, df_precursors

    df_total = pd.read_csv(df_path)
    df_total.dropna(subset=["fosc1", "BB"], inplace=True)
    df_total = df_total[df_total["fosc1"] > 0]
    df_total = df_total[df_total["fosc1"] < 11]
    df_total["target"] = (
        -np.abs(df_total["ionisation potential (eV)"].values - 5.5)
        - 1 * np.abs(df_total["fosc1"].values - 10)
        - 0.5 * np.abs(df_total["ES1"].values - 3)
    )
    df_total, df_precursors = prepare_df_for_plot(
        df_total, features_frag=features_frag
    )
    return df_total, df_precursors


def save_data(df_total):
    import datetime

    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d")
    df_total.to_csv(f"data/output/Full_datatset/df_total_{now}.csv")


def featurise(X, keys):
    """
    Function that featurizes molecules.

    X: Input the values of the array in the dataframe
    keys: list of InchIkeys of the 6mer
    """
    features = []
    m, n = X.shape

    for i in tqdm(range(m), desc="Featurizing molecules"):
        feature = np.zeros(m)
        for j in range(n):
            mol = X[i, j]
            if mol is not None:
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                fingerprint_array = np.array(list(map(int, fingerprint.ToBitString())))
                feature[0:len(fingerprint_array)] = fingerprint_array
        features.append(feature)

    features = np.array(features)

    # # Remove columns with NaN values
    # features = features[:, ~np.isnan(features).any(axis=0)]

    # Remove columns with zero variance
    selector = VarianceThreshold(threshold=0.0)
    features = selector.fit_transform(features)

    # # save the inchikeys of the successfully featurised molecules
    # keys = np.array(keys)
    # keys = keys[~np.isnan(features).any(axis=0)]
    
    features_tensor = torch.tensor(features, dtype=torch.float32)
    print('size of the features tensor:', features_tensor.size())

    return features_tensor

def main():
    df_path = '/rds/general/user/cb1319/home/GEOM3D/STK_path/data/output/Full_dataset/df_total_subset_16_11_23.csv'
    df_precursors_path = '/rds/general/user/cb1319/home/GEOM3D/STK_path/data/output/Prescursor_data/calculation_data_precursor_071123_clean.pkl'

    df_total, df_precursors = load_data_from_file(df_path, df_precursors_path)

    y_IP = df_total['ionisation potential (eV)'].values
    X_6mer_inch = df_total['BB'].values
    X_frag_mol = df_precursors['mol_opt'].values
    X_frag_inch = df_precursors['InChIKey'].values
    keys_6mer = df_total['InChIKey'].values

    print('Number of Oligomers in the dataset:', len(keys_6mer))

    # change the code above so that the length of the array is the same as the number of molecules in the dataset, and the ones that fail to convert to molecules are set to None
    X_6mer_mol = [[] for _ in range(6)]
    inchkey_to_molecule = dict(zip(X_frag_inch, X_frag_mol))
    conversion_fail = 0

    for i in np.arange(0, 6, 1):
        mol_list = []
        temp_list = df_total[f'InChIKey_{i}'].values
        for j in range(len(temp_list)):
            inchkey = temp_list[j]
            if inchkey in inchkey_to_molecule:
                mol_list.append(inchkey_to_molecule[inchkey])
            else:
                conversion_fail += 1
                mol_list.append(None)
                
        X_6mer_mol[i] = mol_list

    max_molecules = max(len(position) for position in X_6mer_mol)
    X_6mer_array = np.full((max_molecules, 6), None, dtype=object)

    for i, position in enumerate(X_6mer_mol):
        X_6mer_array[:len(position), i] = position

    print('X_6mer_array shape:', X_6mer_array.shape)


    # morgan_fingerprints, featurised_keys = featurise(X_6mer_array, keys_6mer)
    morgan_fingerprints = featurise(X_6mer_array, keys_6mer)

    # # make a new empty column in the dataframe
    # df_total['Morgan_fingerprints'] = None

    # # go through the df_total['InChIKey'].values and if the InChIKey in the list of featurised_keys, then append the corresponding row to the column, if there is no corresponding row, then append None
    # for i in range(len(df_total['InChIKey'].values)):
    #     if df_total['InChIKey'].values[i] in featurised_keys:
    #         df_total['Morgan_fingerprints'].values[i] = morgan_fingerprints[featurised_keys == df_total['InChIKey'].values[i]]
    
    # save the morgan fingerprints to the dataframe
    df_total['Morgan_fingerprints'] = morgan_fingerprints.numpy().tolist()
    
    # save the dataframe with the Morgan fingerprints
    df_total.to_csv(df_path, index=False)

    print('Finished featurising molecules and saved the dataframe with Morgan fingerprints')

if __name__ == "__main__":
    main()
