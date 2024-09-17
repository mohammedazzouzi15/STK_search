"""plot the performance of the prediction model on the new molecules."""

import os
from pathlib import Path

import pandas as pd
import pymongo
import stk
import torch

from stk_search.geom3d import dataloader, oligomer_encoding_with_transformer
from stk_search.utils.config_utils import save_config


def get_dataset_from_df(dataset_all, df, config):
    """Check the input dataset for the oligomer embeddiing model and add missing molecules to the dataset.

    Args:
    ----
        dataset_all: list of dictionaries
            list of dictionaries containing the information of the molecules in the dataset.
        df: pandas dataframe
            dataframe containing the information of the molecules.
        config: dictionary
            dictionary containing the configuration of the model.

    Returns:
    -------
        dataset: list of dictionaries.

    """
    dataset_all_dict = {data["InChIKey"]: data for data in dataset_all}
    dataset = []
    missing_inchikey = []
    for Inchikey in df["InChIKey"]:
        if Inchikey in dataset_all_dict:
            dataset.append(dataset_all_dict[Inchikey])
        else:
            missing_inchikey.append(Inchikey)
    df_missing = df[df["InChIKey"].isin(missing_inchikey)].copy()
    df_missing = df_missing.reset_index(drop=True)
    client = pymongo.MongoClient(config["pymongo_client"])
    db = stk.ConstructedMoleculeMongoDb(
        client,
        database=config["database_name"],
    )
    radius = config["model"].get("cutoff", 0.1)
    dataset_missing = dataloader.generate_dataset(
        df_missing,
        db,
        number_of_molecules=df_missing.shape[0],
        model_name=config["model_name"],
        radius=radius,
    )
    dataset.extend(dataset_missing)
    # update the target
    dataset = update_target_on_dataset(dataset, df, config["target_name"])
    return dataset, dataset_missing


# save the dataset for the transformer model if already calculated dataset all exists
def get_dataset_frag_from_df(dataset_all_frag, df, config):
    """Check the input dataset for the oligomer encoding model and add missing molecules to the dataset.

    Args:
    ----
        dataset_all_frag: list of dictionaries
            list of dictionaries containing the information of the molecules in the dataset
        df: pandas dataframe
            dataframe containing the information of the molecules
        config: dictionary
            dictionary containing the configuration of the model
    Returns:
    dataset: list of dictionaries

    """
    if len(dataset_all_frag) == 0:
        dataset_all_dict = {}
    else:
        dataset_all_dict = {
            data[0]["InChIKey"]: data for data in dataset_all_frag
        }
    # dataset = [dataset_all_dict[Inchikey] for Inchikey in df['InChIKey']]
    dataset = []
    missing_inchikey = []
    for Inchikey in df["InChIKey"]:
        if Inchikey in dataset_all_dict:
            dataset.append(dataset_all_dict[Inchikey])
        else:
            missing_inchikey.append(Inchikey)
    df_missing = df[df["InChIKey"].isin(missing_inchikey)].copy()
    df_missing = df_missing.reset_index(drop=True)

    dataset_missing, _ = dataloader.load_data_frag(
        config, df_total=df_missing, dataset_name="missing"
    )
    dataset.extend(dataset_missing)
    return dataset


def update_dataset_learned_embedding(
    df, dataset_all_frag, config, extension="all"
):
    """Check the input dataset for the learned embedding model and add missing molecules to the dataset.

    Args:
    ----
        df: pandas dataframe
            dataframe containing the information of the molecules
        config: dictionary
            dictionary containing the configuration of the model
    Returns:
        dataset: list of dictionaries

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_learned_embedding = torch.load(
        config[f"dataset_learned_embedding_{extension}"], map_location=device
    )
    dataset_all_dict = {
        data["InChIKey"]: data for data in dataset_learned_embedding
    }
    dataset_learned_embedding_update = []
    missing_inchikey = []
    for Inchikey in df["InChIKey"]:
        if Inchikey in dataset_all_dict:
            dataset_learned_embedding_update.append(dataset_all_dict[Inchikey])
        else:
            missing_inchikey.append(Inchikey)
    df_missing = df[df["InChIKey"].isin(missing_inchikey)].copy()
    df_missing = df_missing.reset_index(drop=True)
    dataset_frag_dict = {
        data[0]["InChIKey"]: data for data in dataset_all_frag
    }
    dataset_frag_missing = []
    for Inchikey in missing_inchikey:
        if Inchikey in dataset_frag_dict:
            dataset_frag_missing.append(dataset_frag_dict[Inchikey])
    ephemeral_dir = (
        config["ephemeral_path"] + f"/{config['name'].replace('_','/')}/"
    )
    dataset_learned_embedding_missing = (
        oligomer_encoding_with_transformer.save_encoding_dataset(
            dataset_frag_missing,
            config,
            dataset_name="_missing",
            save_folder=ephemeral_dir,
        )
    )
    dataset_learned_embedding_update.extend(dataset_learned_embedding_missing)

    return dataset_learned_embedding_update


def update_dataset_oligomer(dataset_path, df_total, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = torch.load(dataset_path, map_location=device)
    dataset, dataset_missing = get_dataset_from_df(dataset, df_total, config)
    torch.save(dataset, dataset_path)
    return dataset, dataset_missing


def update_dataset_frag(dataset_path, df_total, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(dataset_path, map_location=device)
    dataset = get_dataset_frag_from_df(dataset, df_total, config)
    torch.save(dataset, dataset_path)
    return dataset


# save the dataset for the embedding model if already calculated dataset all exists
def save_datasets(config, dataset_train, dataset_val, dataset_test):
    name = config["name"]
    config_dir = config["running_dir"]

    ephemeral_dir = config["ephemeral_path"] + f"/{name.replace('_','/')}/"
    Path(ephemeral_dir).mkdir(parents=True, exist_ok=True)

    torch.save(dataset_train, ephemeral_dir + "dataset_train.pth")
    torch.save(dataset_val, ephemeral_dir + "dataset_val.pth")
    torch.save(dataset_test, ephemeral_dir + "dataset_test.pth")
    config["dataset_path" + "_train"] = ephemeral_dir + "dataset_train.pth"
    config["dataset_path" + "_val"] = ephemeral_dir + "dataset_val.pth"
    config["dataset_path" + "_test"] = ephemeral_dir + "dataset_test.pth"
    save_config(config, config_dir)


def save_datasets_frag(config, dataset_train, dataset_val, dataset_test):
    name = config["name"]
    config_dir = config["running_dir"]

    ephemeral_dir = (
        config["ephemeral_path"] + f"/{name.replace('_','/')}/transformer/"
    )
    Path(ephemeral_dir).mkdir(parents=True, exist_ok=True)
    torch.save(dataset_train, ephemeral_dir + "frag_dataset_train.pth")
    torch.save(dataset_val, ephemeral_dir + "frag_dataset_val.pth")
    torch.save(dataset_test, ephemeral_dir + "frag_dataset_test.pth")
    config["frag_dataset_path" + "_train"] = (
        ephemeral_dir + "frag_dataset_train.pth"
    )
    config["frag_dataset_path" + "_val"] = (
        ephemeral_dir + "frag_dataset_val.pth"
    )
    config["frag_dataset_path" + "_test"] = (
        ephemeral_dir + "frag_dataset_test.pth"
    )
    save_config(config, config_dir)


def update_target_on_dataset(dataset, df, target_name):
    """Update the dataset to have the y as the target property."""
    df = df.copy()
    df.index = df["InChIKey"]
    if target_name not in df.columns:
        return dataset
    for data in dataset:
        data.y = float(df[target_name][data.InChIKey])
    return dataset


def save_datasets_for_training(config):
    """Save the datasets for the embedding and the frag model.

    Args:
    ----
        config: dictionary
            dictionary containing the configuration of the model
    Returns:
        config: dictionary
            dictionary containing the configuration of the model

    """
    config_dir = config["running_dir"]

    df_train = pd.read_csv(config_dir + "df_train.csv", low_memory=False)
    df_val = pd.read_csv(config_dir + "df_val.csv", low_memory=False)
    df_test = pd.read_csv(config_dir + "df_test.csv")
    if os.path.isfile(config["dataset_all_path"]):
        dataset_all = torch.load(
            config["dataset_all_path"], map_location=config["device"]
        )
    else:
        dataset_all = []
    dataset_train, _ = get_dataset_from_df(dataset_all, df_train, config)
    dataset_val, _ = get_dataset_from_df(dataset_all, df_val, config)
    dataset_test, _ = get_dataset_from_df(dataset_all, df_test, config)

    save_datasets(config, dataset_train, dataset_val, dataset_test)

    if os.path.isfile(config["dataset_all_frag_path"]):
        dataset_all_frag = torch.load(
            config["dataset_all_frag_path"], map_location=config["device"]
        )
    else:
        dataset_all_frag = []
    dataset_train_frag = get_dataset_frag_from_df(
        dataset_all_frag, df_train, config
    )
    dataset_val_frag = get_dataset_frag_from_df(
        dataset_all_frag, df_val, config
    )
    dataset_test_frag = get_dataset_frag_from_df(
        dataset_all_frag, df_test, config
    )
    save_datasets_frag(
        config, dataset_train_frag, dataset_val_frag, dataset_test_frag
    )

    return config
