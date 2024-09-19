"""script with the data loading functions.

contains functions to load the data from the database and the dataframe
and to generate the dataset and the dataloader.

"""

from pathlib import Path

import numpy as np
import pandas as pd
import pymongo
import stk
import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from stk_search.geom3d.pl_model import Pymodel, model_setup


def load_data_df(config, df_oligomer, dataset_name="train"):
    """Load the data from the dataframe and the database.

    Args:
    ----
        config: dict
            configuration file.
        df_oligomer: pd.DataFrame
            dataframe of the oligomer dataset.
        dataset_name: name added to the dataset
            dataframe of the precursors.

    Returns:
    -------
        dataset: list
            list of the dataset.

    to do: add the option to load the dataset from a global dataset file

    """
    if (
        f"dataset_path_{dataset_name}" in config
        and Path(config[f"dataset_path_{dataset_name}"]).exists()
    ):
        if "device" in config:
            dataset = torch.load(
                config["dataset_path"], map_location=config["device"]
            )
        else:
            dataset = torch.load(config["dataset_path"])
        return dataset

    client = pymongo.MongoClient(config["pymongo_client"])
    db = stk.ConstructedMoleculeMongoDb(
        client,
        database=config["database_name"],
    )
    radius = config["model"].get("cutoff", 0.1)
    dataset = generate_dataset(
        df_oligomer,
        db,
        number_of_molecules=config["num_molecules"],
        model_name=config["model_name"],
        radius=radius,
    )

    # where the new dataset daves
    if config["save_dataset"]:
        name = config["name"]
        Path(name).mkdir(parents=True, exist_ok=True)
        torch.save(dataset, "training/" + name + f"/{len(dataset)}dataset.pt")
    return dataset


def generate_dataset(
    df_total,
    db,
    number_of_molecules=500,
    model_name="SCHNET",
    radius=0,
    target_name="target",
):
    """Generate the dataset from the dataframe and the database.

    Args:
    ----
        df_total: pd.DataFrame
            dataframe of the oligomer dataset.
        db: stk.ConstructedMoleculeMongoDb
            database of the molecules.
        number_of_molecules: int
            number of molecules to generate.
        model_name: str
            name of the model.
        radius: float
            radius for the graph.
        target_name: str
            name of the target value.

    Returns:
    -------
        data_list: list
            list of the dataset

    """
    number_of_molecules = min(number_of_molecules, len(df_total))
    molecule_index = np.random.choice(
        len(df_total), number_of_molecules, replace=False
    )
    data_list = []
    for i in molecule_index:
        molecule = load_molecule(
            df_total["InChIKey"].iloc[i], df_total[target_name].iloc[i], db
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
    """Load a molecule from the database.

    Args:
    ----
    InChIKey (str):
        the InChIKey of the molecule.
    target (float):
        the target value of the molecule.
    db (stk.ConstructedMoleculeMongoDb):
        the stk constructed molecule database to load the molecule from.

    Returns:
    -------
    - molecule (Data): the molecule as a Data object

    """
    import contextlib

    polymer = None
    with contextlib.suppress(KeyError):
        polymer = db.get({"InChIKey": InChIKey})

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
        )
    return None


def load_frag_dataset_from_file(config, dataset_name="train"):
    """Load the fragment dataset from the file.

    Args:
    ----
        config: dict
            configuration file.
        dataset_name: str
            name of the dataset.

    Returns:
    -------
        dataset: list
            list of the dataset.

    """
    if "device" in config:
        dataset = torch.load(
            config[f"frag_dataset_path_{dataset_name}"],
            map_location=config["device"],
        )
    else:
        dataset = torch.load(
            config[f"frag_dataset_path_{dataset_name}"],
            map_location=config["device"],
        )

    if Path(config["model_embedding_chkpt"]).exists():
        chkpt_path = config["model_embedding_chkpt"]
        checkpoint = torch.load(chkpt_path, map_location=config["device"])
        model, graph_pred_linear = model_setup(config)
        # Pass the model and graph_pred_linear to the Pymodel constructor
        pymodel = Pymodel(model, graph_pred_linear, config)
        # Load the state dictionary
        pymodel.load_state_dict(state_dict=checkpoint["state_dict"])
        pymodel.freeze()
        pymodel.to(config["device"])

        return dataset, pymodel
    return None, None


def load_data_frag(
    config, df_total=None, dataset_opt=None, dataset_name="train"
):
    """Load the fragment dataset from the dataframe or the database.

    Args:
    ----
        config: dict
            configuration file.
        df_total: pd.DataFrame
            dataframe of the oligomer dataset.
        dataset_opt: list
            list of the dataset.
        dataset_name: str
            name of the dataset.

    Returns:
    -------
        dataset: list
            list of the dataset.
        pymodel: Pymodel
            the model to use, this should have be a pymodel as defined in the pymodel class.

    """
    if (
        "frag_dataset_path_" + dataset_name in config
        and Path(config["frag_dataset_path_" + dataset_name]).exists()
    ):
        dataset, model = load_frag_dataset_from_file(config, dataset_name)
        return dataset, model
    if Path(config["model_embedding_chkpt"]).exists():
        if "device" in config:
            dataset_opt = torch.load(
                config["dataset_path"], map_location=config["device"]
            )
        else:
            dataset_opt = torch.load(config["dataset_path"])
    else:
        pass

    if dataset_opt is None:
        generate_function = generate_dataset_frag_pd
    else:
        generate_function = generate_dataset_frag_dataset
        df_total = dataset_opt
    client = pymongo.MongoClient(config["pymongo_client"])
    db = stk.ConstructedMoleculeMongoDb(
        client,
        database=config["database_name"],
    )
    # check if model is in the path
    if Path(config["model_embedding_chkpt"]).exists():
        chkpt_path = config["model_embedding_chkpt"]
        checkpoint = torch.load(chkpt_path, map_location=config["device"])
        model, graph_pred_linear = model_setup(config)
        # Pass the model and graph_pred_linear to the Pymodel constructor
        pymodel = Pymodel(model, graph_pred_linear, config)
        # Load the state dictionary
        pymodel.load_state_dict(state_dict=checkpoint["state_dict"])
        pymodel.to(config["device"])
        model = pymodel.molecule_3D_repr
    else:
        model, graph_pred_linear = model_setup(config)
        # Pass the model and graph_pred_linear to the Pymodel constructor
        pymodel = Pymodel(model, graph_pred_linear, config)
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
    return dataset, pymodel


def generate_dataset_frag_pd(
    df_total,
    model,
    db,
    number_of_molecules=1000,
    number_of_fragement=6,
    device="cuda",
    config=None,
):
    """Generate the fragment dataset from the dataframe.

    Args:
    ----
        df_total: pd.DataFrame
            dataframe of the oligomer dataset.
        model: 3d rpr model
            torch model.
        db: stk.ConstructedMoleculeMongoDb
            database of the molecules.
        number_of_molecules: int
            number of molecules to generate.
        number_of_fragement: int
            number of fragment.
        device: str
            device to use.
        config: dict
            configuration file.

    Returns:
    -------
        data_list: list
            list of the dataset.

    """
    number_of_molecules = min(number_of_molecules, len(df_total))

    molecule_index = np.random.choice(
        len(df_total), number_of_molecules, replace=False
    )
    data_list = []
    radius = config["model"].get("cutoff", 0.1)
    for i in molecule_index:
        moldata = fragment_based_encoding(
            df_total["InChIKey"][i],
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


def generate_dataset_frag_dataset(
    dataset,
    model,
    db,
    number_of_molecules=1000,
    number_of_fragement=6,
    device="cuda",
    config=None,
):
    """Generate the fragment dataset from the dataset.

    Args:
    ----
        dataset: list
            list of the dataset.
        model: 3d rpr model
            torch model.
        db: stk.ConstructedMoleculeMongoDb
            database of the molecules.
        number_of_molecules: int
            number of molecules to generate.
        number_of_fragement: int
            number of fragment.
        device: str
            device to use.
        config: dict
            configuration file.

    Returns:
    -------
        data_list: list
            list of the dataset.

    """
    data_list = []
    number_of_molecules = min(len(dataset), number_of_molecules)
    molecule_index = np.random.choice(
        len(dataset), number_of_molecules, replace=False
    )
    radius = config["model"].get("cutoff", 0.1)
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
    """Encode fragments based on the molecule.

    Args:
    ----
        InChIKey: str
            InChIKey of the molecule.
        db_poly: stk.ConstructedMoleculeMongoDb
            database of the molecules.
        model: 3d rpr model
            torch model.
        number_of_fragement: int
            number of fragment.
        device: str
            device to use.
        model_name: str
            name of the model.
        radius: float
            radius for the graph.

    Returns:
    -------
        frags: list
            list of the fragments.

    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else torch.device("cpu")
    polymer = db_poly.get({"InChIKey": InChIKey})
    frags = []
    dat_list = list(polymer.get_atomic_positions())
    positions = np.vstack(dat_list)
    positions = torch.tensor(positions, dtype=torch.float, device=device)
    atom_types = [
        atom.get_atom().get_atomic_number()
        for atom in polymer.get_atom_infos()
    ]
    atom_types = torch.tensor(atom_types, dtype=torch.long, device=device)
    molecule = Data(x=atom_types, positions=positions, device=device)
    if model_name == "PaiNN" and molecule is not None:
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
            atom_types = [
                atom.get_atomic_number() for atom in molecule_bb.get_atoms()
            ]
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
            if model_name == "PaiNN" and molecule_frag is not None:
                radius_edge_index = radius_graph(
                    molecule_frag.positions, r=radius, loop=False
                )
                molecule_frag.radius_edge_index = radius_edge_index
            frags.append(molecule_frag)
        return frags
    return None


def updata_frag_dataset(frag_dataset, dataset, model, model_name):
    """Update the fragment dataset.

    Args:
    ----
        frag_dataset: list
            list of the fragment dataset.
        dataset: list
            list of the dataset.
        model: torch model
            the model for the prediction.
        model_name: str
            the name of the model.

    Returns:
    -------
        frag_dataset: list
            list of the fragment dataset.

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


def updata_dataset(dataset, df, target_name):
    """Update the oligomer dataset.

    Args:
    ----
        dataset: list
            list of the dataset.
        df: pd.DataFrame
            dataframe of the oligomer dataset.
        target_name: str
            name of the target value.

    Returns:
    -------
        frag_dataset: list
            list of the fragment dataset.

    """
    for data in dataset:
        data.y = None

    return dataset


def generate_dataset_and_dataloader(config):
    """Generate the dataset and the dataloader.

    Args:
    ----
        config: dict
            configuration file.

    Returns:
    -------
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

    def get_dataset_dataloader(config, df_name="train")->tuple:
        df_precursors = pd.read_pickle(config["df_precursor"])
        if f"dataset_path_{df_name}" in config and Path(config[f"dataset_path_{df_name}"]).exists():
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
        df_dataset = pd.read_csv(config["running_dir"] + f"/df_{df_name}.csv")
        dataset = load_data_df(config, df_dataset, df_precursors)
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
    """Get the dataloader.

    Args:
    ----
        dataset: list
            list of the dataset.
        config: dict
            configuration file.

    Returns:
    -------
        loader: torch_geometric.loader.DataLoader
            dataloader for the dataset.

    """
    # Set dataloaders
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
