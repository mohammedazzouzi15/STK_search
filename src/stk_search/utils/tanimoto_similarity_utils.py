import stk
from rdkit import Chem

# helper functions to generate fingerprints and calculate similarity
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from stk_search.utils import database_utils
import numpy as np
import torch
import pymongo
from sklearn.decomposition import PCA, KernelPCA
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# Function to generate Morgan fingerprints
def generate_morgan_fingerprints(molecules, radius=2, n_bits=2048):
    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        for mol in molecules
    ]
    return fingerprints


# Function to generate ECFP fingerprints
def generate_ecfp_fingerprints(molecules, radius=2, n_bits=2048):
    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        for mol in molecules
    ]
    return fingerprints


# Function to calculate Tanimoto similarity between fingerprints
def calculate_tanimoto_similarity(fingerprint1, fingerprint2):
    return DataStructs.TanimotoSimilarity(fingerprint1, fingerprint2)


def get_inchi_key(molecule):
    return stk.InchiKey().get_key(molecule)


def get_mol_from_df_single(InChIKey):
    client = "mongodb://ch-atarzia.ch.ic.ac.uk/"

    database = "stk_mohammed_BO"
    collection_name = "BO_exp1"
    client = pymongo.MongoClient(client)
    db_polymer = stk.ConstructedMoleculeMongoDb(
        client,
        database=database,
    )
    mol = db_polymer.get({"InChIKey": InChIKey}).to_rdkit_mol()
    Chem.SanitizeMol(mol)
    return mol


def get_mol_from_df(df, num_mol):
    client = "mongodb://ch-atarzia.ch.ic.ac.uk/"

    database = "stk_mohammed_BO"
    collection_name = "BO_exp1"
    client = pymongo.MongoClient(client)
    db_polymer = stk.ConstructedMoleculeMongoDb(
        client,
        database=database,
    )
    mol_list = []
    for InChIKey in df["InChIKey"].sample(num_mol).values:
        mol = db_polymer.get({"InChIKey": InChIKey}).to_rdkit_mol()
        Chem.SanitizeMol(mol)
        mol_list.append(mol)
    return mol_list


def get_mol_from_res(results, num_initialisation,db_polymer=None,df_mol_dict=None):
    if db_polymer is None:
        client = "mongodb://ch-atarzia.ch.ic.ac.uk/"
        database = "stk_mohammed_BO"
        collection_name = "BO_exp1"
        client = pymongo.MongoClient(client)
        db_polymer = stk.ConstructedMoleculeMongoDb(
            client,
            database=database,
        )
    if df_mol_dict is None:
        df_mol_dict={}
    mol_list_suggested, mol_list_init = [], []
    for InChIKey in results["InchiKey_acquired"][num_initialisation:]:
        if InChIKey in df_mol_dict:
            mol = df_mol_dict[InChIKey]
        else:
            mol = db_polymer.get({"InChIKey": InChIKey}).to_rdkit_mol()
            Chem.SanitizeMol(mol)
            
            df_mol_dict[InChIKey]=mol
        mol_list_suggested.append(mol)
    for InChIKey in results["InchiKey_acquired"][:num_initialisation]:
        if InChIKey in df_mol_dict:
            mol = df_mol_dict[InChIKey]
        else:
            mol = db_polymer.get({"InChIKey": InChIKey}).to_rdkit_mol()
            Chem.SanitizeMol(mol)
            df_mol_dict[InChIKey]=mol
        mol_list_init.append(mol)


    return mol_list_suggested, mol_list_init


def get_tanimoto_similarity(mol_list):
    morgan_fingerprints = generate_morgan_fingerprints(mol_list)

    tanimoto_sim = np.zeros((len(mol_list), len(mol_list)))
    print("Tanimoto similarity (Morgan):")
    for i in range(len(mol_list)):
        for j in range(len(mol_list)):
            tanimoto_sim[i, j] = calculate_tanimoto_similarity(
                morgan_fingerprints[i], morgan_fingerprints[j]
            )
    # plot distribution of the offdiagonal elements
    tanimoto_sim_off_diag = tanimoto_sim[
        ~np.eye(tanimoto_sim.shape[0], dtype=bool)
    ].flatten()
    return tanimoto_sim, tanimoto_sim_off_diag


def plot_similarity_results_elem_suggested(
    search_results, max_iteration=100, min_iteration=50, group_size=10
):
    """plot the similarity of the molecules found in the search space
    search_results: list of dictionaries with the search results
    num_mol: number of molecules to plot
    num_mol_init: number of molecules in the initialisation
    group_size: number of iterations to group together
    return: array of the similarity of the molecules found

    """
    print("Extracting molecules from the search results")
    list_similarity_to_initial = []
    mol_list = []
    for dict_org in search_results:
        dict = dict_org.copy()

        dict.pop("searched_space_df")
        df = pd.DataFrame.from_records(dict)
        df["InChIKey"] = df["InchiKey_acquired"]
        df = df[df["ids_acquired"] < max_iteration]
        df = df[df["ids_acquired"] > min_iteration]
        [mol_list.append(x) for x in get_mol_from_df(df, df.shape[0])]
        # Generate Morgan fingerprints for the dataset
    print("Generating Morgan fingerprints")
    morgan_fingerprints = generate_morgan_fingerprints(mol_list)
    print("Calculating Tanimoto similarity")
    tanimoto_sim = np.zeros((len(mol_list), len(mol_list)))
    print("Tanimoto similarity (Morgan):")
    for i in range(1, len(mol_list)):
        for j in range(i, len(mol_list)):
            tanimoto_sim[i, j] = calculate_tanimoto_similarity(
                morgan_fingerprints[i], morgan_fingerprints[j]
            )
    tanimoto_sim_off_diag = tanimoto_sim[
        ~np.eye(tanimoto_sim.shape[0], dtype=bool)
    ].flatten()
    tanimoto_sim_off_diag = tanimoto_sim_off_diag[tanimoto_sim_off_diag > 0]

    return tanimoto_sim_off_diag


def get_mean_similarity(mol_list_suggested, mol_list_init):
    morgan_fingerprints_suggested = generate_morgan_fingerprints(
        mol_list_suggested
    )
    morgan_fingerprints_init = generate_morgan_fingerprints(mol_list_init)
    tanimoto_sim = np.zeros((len(mol_list_suggested), len(mol_list_init)))
    for i in range(len(mol_list_suggested)):
        for j in range(len(mol_list_init)):
            tanimoto_sim[i, j] = calculate_tanimoto_similarity(
                morgan_fingerprints_suggested[i], morgan_fingerprints_init[j]
            )
    return tanimoto_sim

def moving_average(x, w):
    return np.convolve(x, np.ones(w), "same") / w

def plot_similarity_results_elem_suggested_to_initial(
    search_results,
    nb_iterations=300,
    nb_initialisation=50,
    ax=None,
    color="blue",
    label="",
    df_mol_dict=None,
):
    similarity = np.zeros((len(search_results), nb_iterations))
    nb_iterations_range = np.arange(0, nb_iterations)

    for res_num, dict_org in enumerate(search_results):
        mol_list_suggested, mol_list_init = get_mol_from_res(
            dict_org, num_initialisation=nb_initialisation,df_mol_dict=df_mol_dict
        )

        tanimoto_sim = get_mean_similarity(mol_list_suggested, mol_list_init)
        for i in nb_iterations_range:
            similarity[res_num, i] = np.max(tanimoto_sim[i, :])
        similarity[res_num, 5:-5] = moving_average(similarity[res_num, 5:-5], 5)
            # std_similarity.append(np.std(tanimoto_sim[:i,]))
    mean_similarity = np.mean(similarity, axis=0)
    std_similarity = np.std(similarity, axis=0)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(nb_iterations_range+nb_initialisation, mean_similarity, color=color, label=label)
    ax.fill_between(
        nb_iterations_range+nb_initialisation,
        np.array(mean_similarity) - np.array(std_similarity),
        np.array(mean_similarity) + np.array(std_similarity),
        alpha=0.3,
        color=color
    )
    ax.set_ylabel("max tanimoto similarity \n to initial population")
    ax.set_xlabel("iteration")
    ax.set_ylim(0, 1)
    return ax,df_mol_dict


def plot_similarity_results_elem_suggested_df(
    search_results,
    nb_iterations=300,
    nb_initialisation=50,
    ax=None,
    color="blue",
    label="",
    df_mol_dict=None,
):

    similarity = np.zeros((len(search_results), nb_iterations))
    nb_iterations_range = np.arange(0, nb_iterations)
    client = "mongodb://ch-atarzia.ch.ic.ac.uk/"
    database = "stk_mohammed_BO"
    collection_name = "BO_exp1"
    client = pymongo.MongoClient(client)
    db_polymer = stk.ConstructedMoleculeMongoDb(
        client,
        database=database,
    )
    for res_num, dict_org in enumerate(search_results):

        mol_list_suggested, mol_list_init = get_mol_from_res(
            dict_org, num_initialisation=nb_initialisation,db_polymer=db_polymer,
            df_mol_dict=df_mol_dict
        )

        tanimoto_sim = get_mean_similarity(mol_list_suggested, mol_list_suggested)
        for i in nb_iterations_range:
            sim_matrix = tanimoto_sim[: i + 1, :i+1]
            sim_matrix = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)].flatten()
            similarity[res_num, i] = np.mean(sim_matrix)
            # std_similarity.append(np.std(tanimoto_sim[:i,])
      
    mean_similarity = np.mean(similarity, axis=0)
    std_similarity = np.std(similarity, axis=0)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(nb_iterations_range+nb_initialisation, mean_similarity, color=color, label=label)
    ax.fill_between(
        nb_iterations_range+nb_initialisation,
        np.array(mean_similarity) - np.array(std_similarity),
        np.array(mean_similarity) + np.array(std_similarity),
        alpha=0.3,
        color=color
    )
    ax.set_ylabel("mean tanimoto similarity \n of oligomers suggested")
    ax.set_xlabel("iteration")
    ax.set_ylim(0, 1)
    return ax,df_mol_dict
