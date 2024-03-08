"""
script with the data loading functions
created by Mohammed Azzouzi
date: 2023-11-14
"""

import stk
import pymongo
import numpy as np
import os
import pandas as pd
import torch.optim as optim
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import torch.nn.functional as Functional
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
from stk_search.utils import database_utils
from stk_search.geom3d.pl_model import Pymodel, model_setup
from torch_cluster import radius_graph


def load_data_df(config, df_oligomer, dataset_name="train"):
    """Load the data from the dataframe and the database
    Args:
        config: dict
            configuration file
        df_oligomer: pd.DataFrame
            dataframe of the oligomer dataset
        dataset_name: name added to the dataset
            dataframe of the precursors
    Returns:
        dataset: list
            list of the dataset

    to do: add the option to load the dataset from a global dataset file
    """

    if f"dataset_path_{dataset_name}" in config.keys():
        if os.path.exists(config[f"dataset_path_{dataset_name}"]):
            if "device" in config.keys():
                dataset = torch.load(
                    config["dataset_path"], map_location=config["device"]
                )
            else:
                dataset = torch.load(config["dataset_path"])
            return dataset
        else:
            print("dataset not found")

    client = pymongo.MongoClient(config["pymongo_client"])
    db = stk.ConstructedMoleculeMongoDb(
        client,
        database=config["database_name"],
    )
    radius = config["model"]["cutoff"] if "cutoff" in config["model"] else 0.1
    dataset = generate_dataset(
        df_oligomer,
        db,
        number_of_molecules=config["num_molecules"],
        model_name=config["model_name"],
        radius=radius,
    )

    print(f"length of dataset: {len(dataset)}")

    # where the new dataset daves
    if config["save_dataset"]:
        name = config["name"]
        os.makedirs(name, exist_ok=True)
        torch.save(dataset, "training/" + name + f"/{len(dataset)}dataset.pt")
        print(f"dataset saved to {name}/{len(dataset)}dataset.pt")
    return dataset


def generate_dataset(
    df_total,
    db,
    number_of_molecules=500,
    model_name="SCHNET",
    radius=0,
    target_name="target",
):
    """Generate the dataset from the dataframe and the database

    Args:
        df_total: pd.DataFrame
            dataframe of the oligomer dataset
        db: stk.ConstructedMoleculeMongoDb
            database of the molecules
        number_of_molecules: int
            number of molecules to generate
    Returns:
        data_list: list
            list of the dataset

    """

    if number_of_molecules > len(df_total):
        number_of_molecules = len(df_total)
        print(
            "Number of molecules is greater than the number of molecules in the dataset",
            number_of_molecules,
            len(df_total),
        )
    molecule_index = np.random.choice(
        len(df_total), number_of_molecules, replace=False
    )
    data_list = []
    for i in molecule_index:
        # try:
        #     molecule = load_molecule(
        #         df_total["InChIKey"][i], df_total["target"][i], db
        #     )
        #     data_list.append(molecule)
        # except KeyError:
        #     print(f"No key found in the database for molecule at index {i}")
        molecule = load_molecule(
            df_total["InChIKey"][i], df_total[target_name][i], db
        )
        if model_name == "PaiNN":
            if molecule is not None:
                radius_edge_index = radius_graph(
                    molecule.positions, r=radius, loop=False
                )
                molecule.radius_edge_index = radius_edge_index
                data_list.append(molecule)
        else:
            data_list.append(molecule)
    return data_list


def load_molecule(InChIKey, target, db):
    """
    Load a molecule from the database

    Args:
    - InChIKey (str): the InChIKey of the molecule
    - target (float): the target value of the molecule
    - db (stk.ConstructedMoleculeMongoDb): the database

    Returns:
    - molecule (Data): the molecule as a Data object
    """
    polymer = None
    try:
        polymer = db.get({"InChIKey": InChIKey})
        # Print the complete dictionary returned from the database
        # print("Database entry for InChIKey:", polymer)
    except KeyError:
        print(f"No key found in the database with a key of: {InChIKey}")
        # Handle the missing key case (e.g., return a default value or raise an exception)

    if polymer is not None:
        dat_list = list(polymer.get_atomic_positions())
        positions = np.vstack(dat_list)
        positions = torch.tensor(positions, dtype=torch.float)
        atom_types = list(
            [
                atom.get_atom().get_atomic_number()
                for atom in polymer.get_atom_infos()
            ]
        )
        atom_types = torch.tensor(atom_types, dtype=torch.long)
        y = torch.tensor(target, dtype=torch.float32)

        molecule = Data(
            x=atom_types, positions=positions, y=y, InChIKey=InChIKey
        )
        return molecule
    else:
        return None


def load_frag_dataset_from_file(config, dataset_name="train"):
    """Load the fragment dataset from the file
    Args:
        config: dict
            configuration file
    Returns:
        dataset: list
            list of the dataset
    """
    print(
        f"loading dataset from {config[f'frag_dataset_path_{dataset_name}']}"
    )
    if "device" in config.keys():
        dataset = torch.load(
            config[f"frag_dataset_path_{dataset_name}"],
            map_location=config["device"],
        )
    else:
        dataset = torch.load(
            config[f"frag_dataset_path_{dataset_name}"],
            map_location=config["device"],
        )

    if os.path.exists(config["model_embedding_chkpt"]):
        chkpt_path = config["model_embedding_chkpt"]
        checkpoint = torch.load(chkpt_path, map_location=config["device"])
        model, graph_pred_linear = model_setup(config)
        print("Model loaded: ", config["model_name"])
        # Pass the model and graph_pred_linear to the Pymodel constructor
        pymodel = Pymodel(model, graph_pred_linear, config)
        # Load the state dictionary
        pymodel.load_state_dict(state_dict=checkpoint["state_dict"])
        pymodel.freeze()
        pymodel.to(config["device"])

        return dataset, pymodel
    else:
        print("model not found and ")
        return None, None


def load_data_frag(
    config, df_total=None, dataset_opt=None, dataset_name="train"
):
    """Load the fragment dataset from the dataframe or the database
    Args:
        config: dict
            configuration file
        df_total: pd.DataFrame
            dataframe of the oligomer dataset
        dataset_opt: list
            list of the dataset
    Returns:
        dataset: list
            list of the dataset
        pymodel: Pymodel
            the model"""
    if "frag_dataset_path_" + dataset_name in config.keys():
        if os.path.exists(config["frag_dataset_path_" + dataset_name]):
            dataset, model = load_frag_dataset_from_file(config, dataset_name)
            return dataset, model
        else:
            print("dataset frag not found")
    if os.path.exists(config["dataset_path"]):
        if "device" in config.keys():
            dataset_opt = torch.load(
                config["dataset_path"], map_location=config["device"]
            )
        else:
            dataset_opt = torch.load(config["dataset_path"])
    else:
        print("opt dataset not found")

    if dataset_opt is None:
        print("loading dataset from dataframe")
        generate_function = generate_dataset_frag_pd
    else:
        print("loading dataset from org dataset")
        generate_function = generate_dataset_frag_dataset
        df_total = dataset_opt
    client = pymongo.MongoClient(config["pymongo_client"])
    db = stk.ConstructedMoleculeMongoDb(
        client,
        database=config["database_name"],
    )
    # check if model is in the path
    if os.path.exists(config["model_embedding_chkpt"]):
        chkpt_path = config["model_embedding_chkpt"]
        checkpoint = torch.load(chkpt_path, map_location=config["device"])
        model, graph_pred_linear = model_setup(config)
        print("Model loaded: ", config["model_name"])
        # Pass the model and graph_pred_linear to the Pymodel constructor
        pymodel = Pymodel(model, graph_pred_linear, config)
        # Load the state dictionary
        pymodel.load_state_dict(state_dict=checkpoint["state_dict"])
        pymodel.to(config["device"])
        model = pymodel.molecule_3D_repr
        dataset = generate_function(
            df_total,
            model,
            db,
            number_of_molecules=config["num_molecules"],
            number_of_fragement=config["number_of_fragement"],
            device=config["device"],
            config=config,
        )
        # if config["save_dataset_frag"]:
        #   name = config["name"] + "/transformer"
        #  os.makedirs(name, exist_ok=True)
        # torch.save(dataset, name + "/dataset_frag.pt")
        return dataset, pymodel
    else:
        print("model not found")
        return None, None


def generate_dataset_frag_pd(
    df_total,
    model,
    db,
    number_of_molecules=1000,
    number_of_fragement=6,
    device="cuda",
    config=None,
):
    """Generate the fragment dataset from the dataframe
    Args:
        df_total: pd.DataFrame
            dataframe of the oligomer dataset
        model: 3d rpr model
            torch model
        db: stk.ConstructedMoleculeMongoDb
            database of the molecules
        number_of_molecules: int
            number of molecules to generate
        number_of_fragement: int
            number of fragment
        device: str
            device to use
    Returns:
        data_list: list
            list of the dataset
    """

    molecule_index = np.random.choice(
        len(df_total), number_of_molecules, replace=False
    )
    data_list = []
    for i in molecule_index:
        moldata = fragment_based_encoding(
            df_total["InChIKey"][i],
            db,
            model,
            number_of_fragement,
            device=device,
            model_name=config["model_name"],
            radius=config["model"]["cutoff"],
        )
        if moldata is not None:
            data_list.append(moldata)
    return data_list


def generate_dataset_frag_dataset(
    dataset,
    model,
    db,
    number_of_molecules=1000,
    number_of_fragement=6,
    device="cuda",
    config=None,
):
    """Generate the fragment dataset from the dataset
    Args:
        dataset: list
            list of the dataset
        model: 3d rpr model
            torch model
        db: stk.ConstructedMoleculeMongoDb
            database of the molecules
        number_of_molecules: int
            number of molecules to generate
        number_of_fragement: int
            number of fragment
        device: str
            device to use
    Returns:
        data_list: list
            list of the dataset
    """
    data_list = []
    if len(dataset) < number_of_molecules:
        number_of_molecules = len(dataset)
    molecule_index = np.random.choice(
        len(dataset), number_of_molecules, replace=False
    )
    radius = config["model"]["cutoff"] if "cutoff" in config["model"] else 0.1
    for i in molecule_index:
        moldata = fragment_based_encoding(
            dataset[i]["InChIKey"],
            db,
            model,
            number_of_fragement,
            device=device,
            model_name=config["model_name"],
            radius=radius,
        )
        if moldata is not None:
            data_list.append(moldata)
    return data_list


def fragment_based_encoding(
    InChIKey,
    db_poly,
    model,
    number_of_fragement=6,
    device=None,
    model_name="PaiNN",
    radius=0.1,
):
    """Fragment based encoding
    Args:
        InChIKey: str
            InChIKey of the molecule
        db_poly: stk.ConstructedMoleculeMongoDb
            database of the molecules
        model: 3d rpr model
            torch model
        number_of_fragement: int
            number of fragment
        device: str
            device to use
    Returns:
        frags: list
            list of the fragments
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else torch.device("cpu")
    polymer = db_poly.get({"InChIKey": InChIKey})
    frags = []
    dat_list = list(polymer.get_atomic_positions())
    positions = np.vstack(dat_list)
    positions = torch.tensor(positions, dtype=torch.float, device=device)
    atom_types = list(
        [
            atom.get_atom().get_atomic_number()
            for atom in polymer.get_atom_infos()
        ]
    )
    atom_types = torch.tensor(atom_types, dtype=torch.long, device=device)
    molecule = Data(x=atom_types, positions=positions, device=device)
    if model_name == "PaiNN":
        if molecule is not None:
            radius_edge_index = radius_graph(
                molecule.positions, r=radius, loop=False
            )
            molecule.radius_edge_index = radius_edge_index
    if len(list(polymer.get_building_blocks())) == number_of_fragement:
        with torch.no_grad():
            if model_name == "PaiNN":
                batch = torch.zeros_like(molecule.x)
                opt_geom_encoding = model(
                    molecule.x,
                    molecule.positions,
                    molecule.radius_edge_index,
                    batch,
                )
            else:
                opt_geom_encoding = model(molecule.x, molecule.positions)

        for molecule_bb in polymer.get_building_blocks():
            dat_list = list(molecule_bb.get_atomic_positions())
            positions = np.vstack(dat_list)
            positions = torch.tensor(
                positions, dtype=torch.float, device=device
            )
            atom_types = list(
                [atom.get_atomic_number() for atom in molecule_bb.get_atoms()]
            )
            atom_types = torch.tensor(
                atom_types, dtype=torch.long, device=device
            )
            molecule_frag = Data(
                x=atom_types,
                positions=positions,
                device=device,
                y=opt_geom_encoding,
                InChIKey=InChIKey,
            )
            if model_name == "PaiNN":
                if molecule_frag is not None:
                    radius_edge_index = radius_graph(
                        molecule_frag.positions, r=radius, loop=False
                    )
                    molecule_frag.radius_edge_index = radius_edge_index
            frags.append(molecule_frag)
        return frags


def updata_frag_dataset(
    frag_dataset, dataset, model, model_name
):
    """Update the fragment dataset
    Args:
        frag_dataset: list
            list of the fragment dataset
        dataset: list
            list of the dataset
        model: torch model
            the model for the prediction
        model_name: str
            the name of the model
    Returns:
        frag_dataset: list
            list of the fragment dataset
    """
    dataset_dict = {data.InChIKey: data for data in dataset}

    for data in frag_dataset:

        data_oligomer = dataset_dict[data[0].InChIKey]
        with torch.no_grad():
            if model_name == "PaiNN":
                batch = torch.zeros_like(data_oligomer.x)
                opt_geom_encoding = model(
                    data_oligomer.x,
                    data_oligomer.positions,
                    data_oligomer.radius_edge_index,
                    batch,
                )
            else:
                opt_geom_encoding = model(
                    data_oligomer.x, data_oligomer.positions
                )

        for i in range(len(data)):
            data[i].y = opt_geom_encoding
    return frag_dataset


def generate_dataset_and_dataloader(config):
    """Generate the dataset and the dataloader
    Args:
        config: dict
            configuration file
    Returns:
        train_loader: torch_geometric.loader.DataLoader
            dataloader for the training set
        val_loader: torch_geometric.loader.DataLoader
            dataloader for the validation set
        test_loader: torch_geometric.loader.DataLoader
            dataloader for the test set
        dataset_train: list
            list of the training dataset
        dataset_val: list
            list of the validation dataset
        dataset_test: list
            list of the test dataset
    """

    def get_dataset_dataloader(config, df_name="train"):
        df_precursors = pd.read_pickle(config["df_precursor"])
        if f"dataset_path_{df_name}" in config.keys():
            if os.path.exists(config["dataset_path" + f"_{df_name}"]):
                if "device" in config.keys():
                    dataset = torch.load(
                        config["dataset_path" + f"_{df_name}"],
                        map_location=config["device"],
                    )
                else:
                    dataset = torch.load(
                        config["dataset_path" + f"_{df_name}"]
                    )
                data_loader = get_data_loader(dataset, config)
                return dataset, data_loader
            else:
                print("dataset not found")
        df = pd.read_csv(config["running_dir"] + f"/df_{df_name}.csv")
        dataset = load_data_df(config, df, df_precursors)
        data_loader = get_data_loader(dataset, config)
        return dataset, data_loader

    dataset_train, train_loader = get_dataset_dataloader(
        config, df_name="train"
    )
    dataset_val, val_loader = get_dataset_dataloader(config, df_name="val")
    dataset_test, test_loader = get_dataset_dataloader(config, df_name="test")

    return (
        train_loader,
        val_loader,
        test_loader,
        dataset_train,
        dataset_val,
        dataset_test,
    )


def get_data_loader(dataset, config):
    """Get the dataloader
    Args:
        dataset: list
            list of the dataset
        config: dict
            configuration file

    Returns:
        loader: torch_geometric.loader.DataLoader
            dataloader for the dataset
    """
    # Set dataloaders
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )

    return loader


## old code


def load_data(config):
    if config["load_dataset"]:
        if os.path.exists(config["dataset_path"]):
            if "device" in config.keys():
                dataset = torch.load(
                    config["dataset_path"], map_location=config["device"]
                )
            else:
                dataset = torch.load(config["dataset_path"])
            return dataset
        else:
            print("dataset not found")
    df_path = Path(
        config["STK_path"], "data/output/Full_dataset/", config["df_total"]
    )
    df_precursors_path = Path(
        config["STK_path"],
        "data/output/Prescursor_data/",
        config["df_precursor"],
    )
    # check if file is a path
    if os.path.isfile(df_path):
        df_total, df_precursors = database_utils.load_data_from_file(
            df_path, df_precursors_path
        )
    else:
        df_total, df_precursors = database_utils.load_data_database(
            df_precursor_loc=df_precursors_path,
            num_fragm=config["number_of_fragement"],
        )
        config["df_total"] = database_utils.save_data(df_total)
    client = pymongo.MongoClient(config["pymongo_client"])
    db = stk.ConstructedMoleculeMongoDb(
        client,
        database=config["database_name"],
    )

    dataset = generate_dataset(
        df_total,
        df_precursors,
        db,
        number_of_molecules=config["num_molecules"],
    )

    print(f"length of dataset: {len(dataset)}")

    # where the new dataset daves
    if config["save_dataset"]:
        name = config["name"]
        os.makedirs(name, exist_ok=True)
        torch.save(dataset, "training/" + name + f"/{len(dataset)}dataset.pt")
        print(f"dataset saved to {name}/{len(dataset)}dataset.pt")
    return dataset


def load_3d_rpr(model, output_model_path):
    saved_model_dict = torch.load(output_model_path)
    model.load_state_dict(saved_model_dict["model"])
    # model.eval()
    # check if the function has performed correctly
    print(model)
    return model


def train_val_split(dataset, config):
    seed = config["seed"]
    num_mols = config["num_molecules"]
    assert num_mols <= len(dataset)
    np.random.seed(seed)
    all_idx = np.random.choice(len(dataset), num_mols, replace=False)
    Ntrain = int(num_mols * config["train_ratio"])
    train_idx = all_idx[:Ntrain]
    valid_idx = all_idx[Ntrain:]
    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    train_dataset = [dataset[x] for x in train_idx]
    valid_dataset = [dataset[x] for x in valid_idx]
    # Set dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    return train_loader, val_loader


def load_data_frag_old(config):
    dataset_opt = None
    if config["load_dataset"]:
        if os.path.exists(config["dataset_path_frag"]):
            print(f"loading dataset from {config['dataset_path_frag']}")
            if "device" in config.keys():
                dataset = torch.load(
                    config["dataset_path_frag"],
                    map_location=config["device"],
                )
            else:
                dataset = torch.load(
                    config["dataset_path_frag"],
                    map_location=config["device"],
                )

            if os.path.exists(config["model_embedding_chkpt"]):
                chkpt_path = config["model_embedding_chkpt"]
                checkpoint = torch.load(
                    chkpt_path, map_location=config["device"]
                )
                model, graph_pred_linear = model_setup(config)
                print("Model loaded: ", config["model_name"])
                # Pass the model and graph_pred_linear to the Pymodel constructor
                pymodel = Pymodel(model, graph_pred_linear)
                # Load the state dictionary
                pymodel.load_state_dict(state_dict=checkpoint["state_dict"])
                pymodel.freeze()
                pymodel.to(config["device"])
                model = pymodel.molecule_3D_repr
                return dataset, model
            else:
                print("model not found")
                return None, None
        else:
            print("dataset frag not found")
        if os.path.exists(config["dataset_path"]):
            if "device" in config.keys():
                dataset_opt = torch.load(
                    config["dataset_path"], map_location=config["device"]
                )
            else:
                dataset_opt = torch.load(config["dataset_path"])
        else:
            print("opt dataset not found")

    if dataset_opt is None:
        df_path = Path(
            config["STK_path"], "data/output/Full_dataset/", config["df_total"]
        )
        df_precursors_path = Path(
            config["STK_path"],
            "data/output/Prescursor_data/",
            config["df_precursor"],
        )

        if os.path.isfile(df_path):
            df_total, df_precursors = database_utils.load_data_from_file(
                df_path, df_precursors_path
            )

        else:
            df_total, df_precursors = database_utils.load_data_database(
                df_precursor_loc=df_precursors_path,
                num_fragm=config["number_of_fragement"],
            )
            config["df_total"] = database_utils.save_data(df_total)

        generate_function = generate_dataset_frag_pd
    else:
        print("loading dataset from org dataset")
        generate_function = generate_dataset_frag_dataset
        df_total = dataset_opt
    client = pymongo.MongoClient(config["pymongo_client"])
    db = stk.ConstructedMoleculeMongoDb(
        client,
        database=config["database_name"],
    )
    # check if model is in the path
    if os.path.exists(config["model_embedding_chkpt"]):
        chkpt_path = config["model_embedding_chkpt"]
        checkpoint = torch.load(chkpt_path, map_location=config["device"])
        model, graph_pred_linear = model_setup(config)
        print("Model loaded: ", config["model_name"])
        # Pass the model and graph_pred_linear to the Pymodel constructor
        pymodel = Pymodel(model, graph_pred_linear)
        # Load the state dictionary
        pymodel.load_state_dict(state_dict=checkpoint["state_dict"])
        pymodel.freeze()
        pymodel.to(config["device"])
        model = pymodel.molecule_3D_repr
        dataset = generate_function(
            df_total,
            model,
            db,
            number_of_molecules=config["num_molecules"],
            number_of_fragement=config["number_of_fragement"],
            device=config["device"],
        )
        if config["save_dataset_frag"]:
            name = config["name"] + "/transformer"
            os.makedirs(name, exist_ok=True)
            torch.save(dataset, name + "/dataset_frag.pt")
        return dataset, model
    else:
        print("model not found")
        return None, None
