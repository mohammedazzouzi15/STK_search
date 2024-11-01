import os
from pathlib import Path

import numpy as np
import pandas as pd
import pymongo
import stk
import torch
from stk_search.utils.config_utils import save_config
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import swifter


def join_keys(polymer):
    keys = [stk.InchiKey().get_key(bb) for bb in polymer.get_building_blocks()]
    return "_".join(keys)


def get_bbs_dict(client, database):
    client = pymongo.MongoClient(client)
    database_name = stk.MoleculeMongoDb(
        client,
        database=database,
    )
    mols = database_name.get_all()
    bbs_dict = {}
    for mol in mols:
        bbs_dict[stk.InchiKey().get_key(mol)] = (
            stk.BuildingBlock.init_from_molecule(
                mol, functional_groups=[stk.BromoFactory()]
            )
        )
    return bbs_dict


def Build_polymers(element: pd.DataFrame, bbs_dict):
    # print(genes)

    InchiKey_cols = [col for col in element.columns if "InChIKey_" in col]
    oligomer_size = len(InchiKey_cols)
    genes = "ABCDEFGH"
    genes = genes[:oligomer_size]
    repeating_unit = ""
    # joins the Genes to make a repeating unit string
    repeating_unit = repeating_unit.join(genes)

    def gen_mol(elem):
        precursors = []
        for fragment in elem[InchiKey_cols].to_numpy().flatten():
            bb = bbs_dict[fragment]
            precursors.append(bb)
        polymer = stk.ConstructedMolecule(
            stk.polymer.Linear(
                building_blocks=precursors,
                repeating_unit=repeating_unit,
                num_repeating_units=1,
                num_processes=1,
            )
        )
        dat_list = list(polymer.get_atomic_positions())
        positions = np.vstack(dat_list)
        positions = torch.tensor(positions, dtype=torch.float)
        atom_types = [
            atom.get_atom().get_atomic_number()
            for atom in polymer.get_atom_infos()
        ]
        atom_types = torch.tensor(atom_types, dtype=torch.long)

        bb_key = join_keys(polymer)
        return Data(
            x=atom_types,
            positions=positions,
            InChIKey=stk.InchiKey().get_key(polymer),
            bb_key=bb_key,
            y=elem["target"],
        )

    element["polymer"] = element.swifter.progress_bar(False).apply(gen_mol, axis=1)
    return element["polymer"].tolist()


def load_molecule(InChIKey, target, db):
    """Load a molecule from the database.

    Args:
    ----
    - InChIKey (str): the InChIKey of the molecule
    - target (float): the target value of the molecule
    - db (stk.ConstructedMoleculeMongoDb): the database

    Returns:
    -------
    - molecule (Data): the molecule as a Data object

    """
    polymer = None
    try:
        polymer = db.get({"InChIKey": InChIKey})
        # Print the complete dictionary returned from the database
        # print("Database entry for InChIKey:", polymer)
    except KeyError:
        pass
        # Handle the missing key case (e.g., return a default value or raise an exception)

    if polymer is not None:
        dat_list = list(polymer.get_atomic_positions())
        positions = np.vstack(dat_list)
        positions = torch.tensor(positions, dtype=torch.float)
        atom_types = [
            atom.get_atom().get_atomic_number()
            for atom in polymer.get_atom_infos()
        ]
        atom_types = torch.tensor(atom_types, dtype=torch.long)
        y = torch.tensor(target, dtype=torch.float32)

        return Data(
            x=atom_types,
            positions=positions,
            y=y,
            InChIKey=InChIKey,
            bb_key=join_keys(polymer),
        )
    else:
        return None


def get_dataset_polymer_opt(config, element):
    client = pymongo.MongoClient(config["pymongo_client"])
    db = stk.ConstructedMoleculeMongoDb(
        client,
        database=config["database_name"],
    )
    element["data_opt"] = element.swifter.progress_bar(False).apply(
        lambda x: load_molecule(x["InChIKey"], x["target"], db), axis=1
    )
    return element["data_opt"].tolist()


def add_position_opt(dataset, dataset_opt):
    for i in range(len(dataset)):
        dataset[i].positions_opt = dataset_opt[i].positions
        dataset[i].x_opt = dataset_opt[i].x
    return dataset


def get_dataset_polymer(element: pd.DataFrame, bbs_dict, config):
    # element_copy = element[[f'InChIKey_{i}' for i in range(oligomer_size)]].copy()
    dataset_poly = Build_polymers(element, bbs_dict)
    dataset_poly_opt = get_dataset_polymer_opt(config, element)
    return add_position_opt(dataset_poly, dataset_poly_opt)


def get_data_loader(dataset, config):
    """Get the dataloader
    Args:
        dataset: list
            list of the dataset
        config: dict
            configuration file.

    Returns
    -------
        loader: torch_geometric.loader.DataLoader
            dataloader for the dataset

    """
    # Set dataloaders
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )


def generate_dataset_and_dataloader(config, bbs_dict):
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
            list of the test dataset.
    """

    def get_dataset_dataloader(config, df_name="train"):
        pd.read_pickle(config["df_precursor"])
        if f"dataset_path_{df_name}" in config:
            if os.path.exists(config["dataset_path" + f"_{df_name}"]):
                if "device" in config:
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
                pass
        df = pd.read_csv(config["running_dir"] + f"/df_{df_name}.csv")
        dataset = get_dataset_polymer(
            element=df, bbs_dict=bbs_dict, config=config
        )
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


def save_datasets(config, dataset_train, dataset_val, dataset_test):
    name = config["name"]
    ephemeral_dir = config["ephemeral_path"] + f"/{name.replace('_','/')}/"
    Path(ephemeral_dir).mkdir(parents=True, exist_ok=True)

    torch.save(dataset_train, ephemeral_dir + "dataset_train.pth")
    torch.save(dataset_val, ephemeral_dir + "dataset_val.pth")
    torch.save(dataset_test, ephemeral_dir + "dataset_test.pth")
    config["dataset_path" + "_train"] = ephemeral_dir + "dataset_train.pth"
    config["dataset_path" + "_val"] = ephemeral_dir + "dataset_val.pth"
    config["dataset_path" + "_test"] = ephemeral_dir + "dataset_test.pth"
    save_config(config, config["running_dir"])
    return config


# df_elements = df_total.sample(500)
# dataset = get_dataset_polymer(
#    oligomer_size=6,
#    element=df_elements,
#    bbs_dict=bbs_dict,
#    config=config
# )
