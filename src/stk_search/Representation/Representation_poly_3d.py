"""Encode the molecules using a pretrained geometry based model.

We use rdkit to generate a first guess of the 3D coordinates of the molecule and then use a pretrained model to encode the Representation of the molecule.
the model has been trained to encode the Representation of the fragments and then use this Representation to encode the Representation of the oligomer.
"""

import numpy as np
import pandas as pd
import pymongo
import stk
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class RepresentationPoly3d:
    """Generate the Representation of the constructed molecules from the fragment representation.

    In this class we use the 3d Representation of the fragements directly as input to the model that will generate the Representation of the oligomer.
    the model has been trained to encode the Representation of the fragments and then use this Representation to encode the Representation of the oligomer.

    Attributes
    ----------
    model_encoding : torch.nn.Module
        The model used to encode the Representation of the fragments.
        the model has been trained to encode the Representation of the fragments and then use this Representation to encode the Representation of the oligomer.
        this model could be any model that takes as input the 3d geometry of the molecule and outputs the Representation of the molecule.
    df_results : pd.dataframe
        The table of building blocks named with their InChIKey.
    data : list
        The list of data containing the Representation of the fragments.
    db_poly : stk.ConstructedMoleculeMongoDb
        The database containing the polymers.
    db_frag : stk.MoleculeMongoDb
        The database containing the fragments.
    device : str
        The device to be used for the encoding.
    oligomer_size : int
        The size of the oligomer.
    mongo_client : str
        The client to connect to the mongodb.
    database : str
        The database to connect to.

    """

    def __init__(
        self,
        model_encoding,
        device=None,
        oligomer_size=6,
        mongo_client="mongodb://ch-atarzia.ch.ic.ac.uk/",
        database="stk_mohammed_new",
    ):
        """Initialise the class.

        Args:
        ----
            model_encoding (torch.nn.Module): model used to encode the Representation of the fragments
            df_results (pd.dataframe): table of building blocks named with their InChIKey
            data (list): list of data containing the Representation of the fragments
            db_poly (stk.ConstructedMoleculeMongoDb): database containing the polymers
            db_frag (stk.MoleculeMongoDb): database containing the fragments
            device (str): device to be used for the encoding
            oligomer_size (int): size of the oligomer
            mongo_client (str): client to connect to the mongodb
            database (str): database to connect to


        """
        if device is None:
            self.device = (
                "cuda" if torch.cuda.is_available() else torch.device("cpu")
            )
        else:
            self.device = device
        self.get_bbs_dict(mongo_client, database)
        self.batch_size = 100
        self.pymodel = model_encoding.to(self.device)
        self.name = "RepresentationPoly3d"
        self.oligomer_size = oligomer_size
        self.dataset_local = {}
        self.dataset_frag_local = {}

    def get_representation_from_dataset(self, df_elements):
        """Get the Representation from the dataset.

        Args:
        ----
            df_elements (pd.dataframe): table of building blocks named with their InChIKey
        Returns:
            torch.tensor: Representation of the constructed molecule
            list: list of InChIKeys to compute

        """
        opt_geom_encoding = []
        keys_to_compute = []
        for key in df_elements["bb_key"]:
            if key in self.dataset_local:
                opt_geom_encoding.append(self.dataset_local[key])
            else:
                keys_to_compute.append(key)
        if len(opt_geom_encoding) > 0:
            opt_geom_encoding = torch.vstack(opt_geom_encoding)
        return opt_geom_encoding, keys_to_compute

    def add_representations_to_dataset(self, df_elements, keys_to_compute):
        """Add the Representation to the dataset.

        Args:
        ----
            df_elements (pd.dataframe): table of building blocks named with their InChIKey
            keys_to_compute (list): list of InChIKeys to compute

        """
        elements_copy = df_elements.loc[list(keys_to_compute)].copy()
        elements_copy = elements_copy.drop_duplicates(subset="bb_key")
        opt_geom_encoding_add = []
        dataset_poly = self.Build_polymers(elements_copy)
        data_loader = self.get_data_loader(dataset_poly)
        opt_geom_encoding_add = [
            self.model_encoding(data) for data in data_loader
        ]
        opt_geom_encoding_add = torch.vstack(opt_geom_encoding_add)
        for ii, bb_key in enumerate(elements_copy["bb_key"]):
            self.dataset_local[bb_key] = opt_geom_encoding_add[ii, :]

    def generate_repr(self, elements):
        """Generate the Representation of the elements.

        Args:
        ----
            elements (pd.dataframe): table of building blocks nmaed with their InChIKey
        Returns:
            torch.tensor: Representation of the constructed molecule

        """
        df_elements = elements.copy()
        df_elements["bb_key"] = df_elements.apply(
            lambda x: self.join_keys_elem(x), axis=1
        )
        df_elements.index = df_elements["bb_key"]
        opt_geom_encoding, keys_to_compute = (
            self.get_representation_from_dataset(df_elements)
        )
        if len(keys_to_compute) > 0:
            self.add_representations_to_dataset(df_elements, keys_to_compute)
            opt_geom_encoding, keys_to_compute = (
                self.get_representation_from_dataset(df_elements)
            )
        if len(keys_to_compute) > 0:
            msg = "Some representations are missing"
            raise ValueError(msg)
        return opt_geom_encoding

    def model_encoding(self, data):
        """Encode the Representation of the molecule.

        Args:
        ----
            data (torch_geometric.data.Data): data containing the Representation of the molecule
            Returns:
            torch.tensor: Representation of the molecule

        """
        pymodel = self.pymodel
        with torch.no_grad():
            return pymodel.transform_to_opt(
                pymodel.molecule_3D_repr(
                    data.x.to(self.device),
                    data.positions.to(self.device),
                    data.batch.to(self.device),
                )
            )

    def get_data_loader(self, dataset):
        """Get the dataloader.

        Args:
        ----
            dataset: list
                list of the dataset
            config: dict
                configuration file.

        Returns:
        -------
            loader: torch_geometric.loader.DataLoader
                dataloader for the dataset

        """
        # Set dataloaders
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def join_keys(self, polymer):
        """Join the keys of the building blocks."""
        keys = [
            stk.InchiKey().get_key(bb) for bb in polymer.get_building_blocks()
        ]
        return "_".join(keys)

    def join_keys_elem(self, element):
        """Join the keys of the building blocks."""
        keys = list(
            element[
                [f"InChIKey_{x}" for x in range(self.oligomer_size)]
            ].values
        )
        return "_".join(keys)

    def get_all(self, database_name):
        for entry in database_name._position_matrices.find():
            # Do 'or' query over all key value pairs.
            query = {
                "$or": [
                    {key: value}
                    for key, value in database_name._get_molecule_keys(entry)
                ]
            }

            json = database_name._molecules.find_one(query)
            if json is None:
                raise KeyError(
                    "No molecule found in the database associated "
                    f"with a position matrix with query: {query}. "
                    "This suggests your database is corrupted."
                )

            yield (
                database_name._dejsonizer.from_json(
                    {
                        "molecule": json,
                        "matrix": entry,
                    }
                ),
                json["InChIKey"],
            )

    def get_bbs_dict(self, client, database):
        """Get the building blocks dictionary."""
        client = pymongo.MongoClient(client)
        database_name = stk.MoleculeMongoDb(
            client,
            database=database,
        )
        mols_keys = self.get_all(database_name)
        bbs_dict = {}
        for mol, key in mols_keys:
            bbs_dict[key] = stk.BuildingBlock.init_from_molecule(
                mol, functional_groups=[stk.BromoFactory()]
            )

        self.bbs_dict = bbs_dict
        return bbs_dict

    def Build_polymers(self, element: pd.DataFrame):
        """Build the polymers.

        Here we first build the polymer from the fragments using stk then we use rdkit etkdg to generate the 3D coordinates of the molecule.
        then we use the model to encode the Representation of the molecule.

        Args:
        ----
            element (pd.DataFrame): table of building blocks named with their InChIKey

        Returns:
        -------
            list: list of data containing the Representation of the constructed molecules
                the data is a torch_geometric.data.Data object containing the Representation of the molecule
                its attibutes are:
                x: atom types
                positions: atomic positions
                InChIKey: InChIKey of the molecule
                bb_key: key of the building blocks

        """
        bbs_dict = self.bbs_dict

        genes = "ABCDEFGH"
        genes = genes[: self.oligomer_size]
        repeating_unit = ""
        # joins the Genes to make a repeating unit string
        repeating_unit = repeating_unit.join(genes)
        InchiKey_cols = [col for col in element.columns if "InChIKey_" in col]  # noqa: N806

        def gen_mol(elem) -> Data:
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

            bb_key = self.join_keys(polymer)
            return Data(
                x=atom_types,
                positions=positions,
                InChIKey=stk.InchiKey().get_key(polymer),
                bb_key=bb_key,
            )

        element["polymer"] = element.apply(
            lambda x: gen_mol(x), axis=1
        )
        return element["polymer"].tolist()
