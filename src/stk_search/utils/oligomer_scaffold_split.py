"""script to turn a a dataset into custom fragment scaffold split."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw, rdFMCS
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from tqdm import tqdm

from stk_search.utils import database_utils


def oligomer_scaffold_splitter(dataset, config):
    """Split a dataset into a training and test set based on the scaffold of the oligomers.

    The test set contains the oligomers with the specified scaffold.

    Args:
    ----
    - dataset: list of dictionaries
    - config: dictionary with the following

    Returns:
    -------
    - test_set_inchikeys: list of InChIKeys of the oligomers in the test set

    """
    df_total, df_precursors = load_dataframes(dataset, config)

    check_data_exists(df_total, dataset, config)

    # Define HDBSCAN parameters (adjust as needed)
    min_cluster_size = config[
        "oligomer_min_cluster_size"
    ]  # Minimum size for a cluster to be considered valid
    min_samples = config[
        "oligomer_min_samples"
    ]  # Minimum number of points required to form a core point

    # Create a HDBSCAN instance
    hdb_model = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    # Fit the model to the average PCA scores from the df_total['2d_tani_pca_1'] and df_total['2d_tani_pca_2']
    cluster_labels = hdb_model.fit_predict(
        df_total[["2d_tani_pca_1", "2d_tani_pca_2"]]
    )
    # assign the cluster labels to the InChIKeys in df_total
    df_total["Cluster"] = cluster_labels
    cluster_assignments = dict(zip(df_total["InChIKey"], df_total["Cluster"]))

    # print the number of oligomers in each cluster
    # print the Number of molecules in each cluster
    chosen_cluster = config[
        "test_set_oligomer_cluster"
    ]  # Choose the cluster you want to use for the test set
    cluster_keys = []
    for key, value in cluster_assignments.items():
        if value == chosen_cluster:
            cluster_keys.append(key)

    return cluster_keys


def cluster_analysis(dataset, config, min_cluster_size=750, min_samples=50):
    """Perform clustering on the dataset and print the number of clusters and the number of oligomers in each cluster.

    Args:
    ----
    - dataset: list of dictionaries
    - config: dictionary with the following
    - min_cluster_size: int, minimum size for a cluster to be considered valid
    - min_samples: int, minimum number of points required to form a core point

    """
    df_total, df_precursors = load_dataframes(dataset, config)
    check_data_exists(df_total, dataset, config)

    # Create a HDBSCAN instance
    hdb_model = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    # Fit the model to the average PCA scores
    hdb_model.fit_predict(df_total[["2d_tani_pca_1", "2d_tani_pca_2"]])

    # print how many clusters there are and how many oligomers are in each cluster


def pca_plot(df_total, config):
    """Plot the 2D PCA space of the dataset, highlighting the chosen cluster.

    Args:
    ----
    - dataset: list of dictionaries
    - config: dictionary with the following

    """
    df_total, df_precursors = load_dataframes(dataset, config)
    check_data_exists(df_total, dataset, config)

    min_cluster_size = config[
        "oligomer_min_cluster_size"
    ]  # Minimum size for a cluster to be considered valid
    min_samples = config[
        "oligomer_min_samples"
    ]  # Minimum number of points required to form a core point

    # Create a HDBSCAN instance
    hdb_model = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    # Fit the model to the average PCA scores
    cluster_labels = hdb_model.fit_predict(
        df_total[["2d_tani_pca_1", "2d_tani_pca_2"]]
    )
    # assign the cluster labels to the InChIKeys in df_total
    df_total["Cluster"] = cluster_labels
    selected_cluster = config[
        "test_set_oligomer_cluster"
    ]  # Choose the cluster you want to use for the test set

    # Plot all clusters
    plt.figure(figsize=(10, 10))
    plt.scatter(
        df_total["2d_tani_pca_1"],
        df_total["2d_tani_pca_2"],
        c=df_total["Cluster"],
        cmap="viridis",
        alpha=0.7,
    )
    # Highlight the specific cluster
    df_cluster_spec = df_total[df_total["Cluster"] == selected_cluster]
    plt.scatter(
        df_cluster_spec["2d_tani_pca_1"],
        df_cluster_spec["2d_tani_pca_2"],
        c="red",
        label=f"Cluster {selected_cluster}",
        alpha=0.9,
    )
    plt.legend()
    plt.title("Clusters of oligomers based on average PCA scores")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


# still to do for oligomer
def substructure_analysis_oligomers(
    dataset, config, selected_cluster=1, min_cluster_size=750, min_samples=50
):
    """Perform substructure analysis for the specified cluster of oligomers.

    Args:
    ----
    - dataset: list of dictionaries
    - config: dictionary with the following
    - selected_cluster: int, the cluster to perform substructure analysis on
    - min_cluster_size: int, minimum size for a cluster to be considered valid
    - min_samples: int, minimum number of points required to form a core point

    """
    df_total, df_precursors = load_dataframes(dataset, config)

    X_frag_mol = df_precursors["mol_opt"].values
    X_frag_inch = df_precursors["InChIKey"].values
    keys_6mer = df_total["InChIKey"].values

    check_data_exists(df_total, dataset, config)

    # Clustering
    min_cluster_size = config["oligomer_min_cluster_size"]
    min_samples = config["oligomer_min_samples"]

    hdb_model = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    cluster_labels = hdb_model.fit_predict(
        df_total[["2d_tani_pca_1", "2d_tani_pca_2"]]
    )
    cluster_assignments = dict(zip(keys_6mer, cluster_labels))

    selected_cluster = selected_cluster
    # Filter out the data points in the specified cluster
    selected_cluster_keys = [
        oligomer_key
        for oligomer_key, cluster_id in cluster_assignments.items()
        if cluster_id == selected_cluster
    ]

    # Generate common substructures for each molecule in the cluster
    common_substructures = []
    counter = 0

    # Loop through the oligomers in the cluster
    for oligomer_key in tqdm(
        selected_cluster_keys,
        desc=f"Generating substructures for Cluster {selected_cluster}",
    ):
        # Extract InChIKeys from columns InChIKeys_0 to InChIKeys_5
        inchikeys = [
            df_total.loc[
                df_total["InChIKey"] == oligomer_key, f"InChIKey_{i}"
            ].values[0]
            for i in range(6)
        ]

        # Get the RDKit molecules for the corresponding InChIKeys
        fragments = [
            X_frag_mol[X_frag_inch == inchikey][0]
            for inchikey in inchikeys
            if inchikey in X_frag_inch
        ]

        # Combine the individual fragments into a single molecule, stepwise because can only take 2 rdkit molecules at a time
        combined_molecule = Chem.CombineMols(fragments[0], fragments[1])
        for i in range(2, len(fragments)):
            combined_molecule = Chem.CombineMols(
                combined_molecule, fragments[i]
            )

        # Convert the combined oligomer molecule to SMILES
        oligomer_smiles = Chem.MolToSmiles(combined_molecule)

        # Check if there's only one molecule in the cluster
        if len(selected_cluster_keys) < 2:
            pass
        else:
            # Find the common substructure in the combined oligomer
            common_substructure = rdFMCS.FindMCS(
                [combined_molecule, combined_molecule]
            )
            common_substructure = Chem.MolFromSmarts(
                common_substructure.smartsString
            )
            common_substructures.append(common_substructure)

        # visualise only one combined molecule in the cluster in 2D, so its easier to see
        if len(fragments) == 6 and counter == 0:
            mol = Chem.MolFromSmiles(oligomer_smiles)
            img = Draw.MolToImage(mol)
            display(img)
            counter += 1

    # Count the occurrences of each substructure
    substructure_counts = Counter(
        [Chem.MolToSmarts(sub) for sub in common_substructures]
    )

    # Rank substructures based on frequency
    ranked_substructures = sorted(
        substructure_counts.items(), key=lambda x: x[1], reverse=True
    )

    # Display the top N substructures and their occurrences
    top_n = min(
        3, len(ranked_substructures)
    )  # Choose the smaller of 3 and the actual number of substructures
    for i, (substructure, _count) in enumerate(ranked_substructures[:top_n]):
        img = Draw.MolToImage(Chem.MolFromSmarts(substructure))
        display(img)


def load_dataframes(dataset, config):
    seed = config["seed"]
    len(dataset)
    np.random.seed(seed)

    df_path = Path(
        config["STK_path"], "data/output/Full_dataset/", config["df_total"]
    )
    df_precursors_path = Path(
        config["STK_path"],
        "data/output/Prescursor_data/",
        config["df_precursor"],
    )

    df_total, df_precursors = database_utils.load_data_from_file(
        df_path, df_precursors_path
    )

    return df_total, df_precursors


def check_data_exists(df_total, dataset, config):
    # check if df_total['2d_tani_pca_1'] and df_total['2d_tani_pca_2'] exist, if not, calculate them
    if (
        "2d_tani_pca_1" in df_total.columns
        and "2d_tani_pca_2" in df_total.columns
    ):
        pass

    else:
        generate_2d_PCA(dataset, config)


def calculate_morgan_fingerprints(mols, radius=2, nBits=1024):
    return [
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        for mol in mols
    ]


def calculate_tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def generate_repr(df_total, df_precursors, frag_properties, idx=0):
    init_rpr = []
    frag_properties = frag_properties.union(["InChIKey"])
    if "InChIKey_0" not in df_total.columns:
        elements_curr = [
            [eval(df_total["BB"][x])[i]["InChIKey"] for i in range(6)]
            for x in idx
        ]
        elements_curr = pd.DataFrame(
            elements_curr, columns=[f"InChIKey_{x}" for x in range(6)]
        )
    else:
        elements_curr = df_total[[f"InChIKey_{x}" for x in range(6)]].iloc[idx]
    num_frag = elements_curr.shape[1]
    init_rpr = []
    for i in range(num_frag):
        elements_curr["InChIKey"] = elements_curr[f"InChIKey_{i}"].astype(str)
        df_eval = pd.merge(
            elements_curr,
            df_precursors[frag_properties],
            on="InChIKey",
            how="left",
            suffixes=("", f"_{i}"),
        )
        if len(init_rpr) == 0:
            init_rpr = df_eval[df_eval.columns[num_frag + 1 :]].values
        else:
            init_rpr = np.concatenate(
                [init_rpr, df_eval[df_eval.columns[num_frag + 1 :]].values],
                axis=1,
            )
    return torch.tensor(np.array(init_rpr.astype(float)), dtype=torch.float32)


def generate_2d_PCA(df, df_precursors):
    """Generate 2D PCA scores for the dataset and append them to df_total."""
    # df_total, df_precursors = load_dataframes(dataset, config)
    df_total = df.copy()
    X_frag_mol = df_precursors["mol_opt"].values

    morgan_fps = calculate_morgan_fingerprints(X_frag_mol)
    tanimoto_sim = np.zeros((len(X_frag_mol), len(X_frag_mol)))
    for i in range(len(X_frag_mol)):
        for j in range(len(X_frag_mol)):
            tanimoto_sim[i, j] = calculate_tanimoto_similarity(
                morgan_fps[i], morgan_fps[j]
            )
            tanimoto_sim[j, i] = tanimoto_sim[i, j]

    # Number of components you want to retain after PCA
    n_components = 7  # You can change this based on your requirements
    # Perform PCA on the Morgan fingerprints
    pca = PCA(n_components=n_components)
    # can chage morgan_fps to tanimoto_sim
    pca_scores = pca.fit_transform(tanimoto_sim)
    # Append PCA scores into 7 new columns in df_precursors
    for i in range(n_components):
        df_precursors[f"PCA_{i}"] = pca_scores[:, i]
    oligomer_pca_scores_2 = generate_repr(
        df_total,
        df_precursors,
        df_precursors.columns[-7:],
        idx=range(len(df_total)),
    )

    # make a dataframe with one column for each 42 pca score and then the first column is the InChIKey
    df_pca_scores = pd.DataFrame(
        oligomer_pca_scores_2, columns=[f"PCA_{i}" for i in range(42)]
    )
    df_pca_scores["InChIKey"] = df_total["InChIKey"]
    # drop the rows with NaN values
    df_pca_scores = df_pca_scores.dropna()
    pca2 = PCA(n_components=2)
    # Perform PCA on the first 42 columns of the dataframe
    oligomer_pca_scores_2 = df_pca_scores[df_pca_scores.columns[:42]].values
    oligomer_pca_scores_2_final = pca2.fit_transform(oligomer_pca_scores_2)

    # append the 2 pca scores to the df_pc_scores dataframe in new columns called 2d_tani_pca_1 and 2d_tani_pca_2
    df_pca_scores["2d_tani_pca_1"] = oligomer_pca_scores_2_final[:, 0]
    df_pca_scores["2d_tani_pca_2"] = oligomer_pca_scores_2_final[:, 1]
    # append the pca scores to the df_total dataframe in new columns called 2d_tani_pca_1 and 2d_tani_pca_2 for the corresponding InChIKey

    df_total["2d_tani_pca_1"] = df_total["InChIKey"].map(
        df_pca_scores.set_index("InChIKey")["2d_tani_pca_1"]
    )
    df_total["2d_tani_pca_2"] = df_total["InChIKey"].map(
        df_pca_scores.set_index("InChIKey")["2d_tani_pca_2"]
    )

    # df_total.to_csv(df_path, index=False)
    # df_precursors.to_csv(df_precursors_path, index=False)

    return df_total, df_precursors
