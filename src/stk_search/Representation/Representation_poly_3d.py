"""
this script is to encode the representation of the oligomer from the representation of the fragments
"""

import numpy as np
import torch
from torch_geometric.data import Data, Batch
import pymongo
from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne
from torch_geometric.loader import DataLoader
import stk
import pandas as pd
import swifter


class Representation_poly_3d:
    def __init__(
        self,
        model_encoding,
        df_results=None,
        data=None,
        db_poly=None,
        db_frag=None,
        device=None,
        oligomer_size=6,
    ):
        """Initialise the class.
        Args:
            model_encoding (torch.nn.Module): model used to encode the representation of the fragments
            df_results (pd.dataframe): table of building blocks named with their InChIKey
            data (list): list of data containing the representation of the fragments
            db_poly (stk.ConstructedMoleculeMongoDb): database containing the polymers
            db_frag (stk.MoleculeMongoDb): database containing the fragments
            device (str): device to be used for the encoding
        """

        if device is None:
            self.device = (
                "cuda" if torch.cuda.is_available() else torch.device("cpu")
            )
        else:
            self.device = device
        # self.model_encoding = model_encoding
        # self.model_encoding.eval()
        # self.model_encoding.to(self.device)
        self.get_bbs_dict(
            "mongodb://ch-atarzia.ch.ic.ac.uk/", "stk_mohammed_new"
        )
        self.batch_size = 100
        self.pymodel = model_encoding.to(self.device)
        self.name = "Representation_poly_3d"
        self.oligomer_size = oligomer_size
        self.dataset_local = {}
        self.dataset_frag_local = {}

    def get_representation_from_dataset(self, df_elements):
        """Get the representation from the dataset.
        Args:
            df_elements (pd.dataframe): table of building blocks named with their InChIKey
        Returns:
            torch.tensor: representation of the constructed molecule
            list: list of InChIKeys to compute
        """
        opt_geom_encoding = []
        keys_to_compute = []
        for key in df_elements["bb_key"]:
            if key in self.dataset_local.keys():
                opt_geom_encoding.append(self.dataset_local[key])
            else:
                keys_to_compute.append(key)
        if len(opt_geom_encoding) > 0:
            opt_geom_encoding = torch.vstack(opt_geom_encoding)
        return opt_geom_encoding, keys_to_compute

    def add_representations_to_dataset(self, df_elements, keys_to_compute):
        """Add the representation to the dataset.
        Args:
            df_elements (pd.dataframe): table of building blocks named with their InChIKey
            keys_to_compute (list): list of InChIKeys to compute
        """
        elements_copy = df_elements.loc[[x for x in keys_to_compute]].copy()
        elements_copy.drop_duplicates(subset="bb_key", inplace=True)
        opt_geom_encoding_add = []
        dataset_poly = self.Build_polymers(elements_copy)
        data_loader = self.get_data_loader(dataset_poly)
        for data in data_loader:
            opt_geom_encoding_add.append(self.model_encoding(data))
        opt_geom_encoding_add = torch.vstack(opt_geom_encoding_add)
        for ii, bb_key in enumerate(elements_copy["bb_key"]):
            self.dataset_local[bb_key] = opt_geom_encoding_add[ii, :]

    def generate_repr(self, elements):
        """Generate the representation of the elements.
        Args:
            elements (pd.dataframe): table of building blocks nmaed with their InChIKey
        Returns:
            torch.tensor: representation of the constructed molecule
        """

        df_elements = elements.copy()
        df_elements["bb_key"] = df_elements.swifter.apply(
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
            raise ValueError("Some representations are missing")
        return opt_geom_encoding

    def model_encoding(self, data):
        """Encode the representation of the molecule.
        Args:
            data (torch_geometric.data.Data): data containing the representation of the molecule
            Returns:
            torch.tensor: representation of the molecule
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
            batch_size=self.batch_size,
            shuffle=False,
            # drop_last=True,
        )

        return loader

    def join_keys(self, polymer):
        keys = [
            stk.InchiKey().get_key(bb) for bb in polymer.get_building_blocks()
        ]
        return "_".join(keys)

    def join_keys_elem(self, element):
        keys = [
            bb
            for bb in element[
                [f"InChIKey_{x}" for x in range(self.oligomer_size)]
            ].values
        ]
        # print(keys)
        return "_".join(keys)

    def get_bbs_dict(self, client, database):
        client = pymongo.MongoClient(client)
        db_mol = stk.MoleculeMongoDb(
            client,
            database=database,
        )
        mols = db_mol.get_all()
        bbs_dict = {}
        for mol in mols:
            bbs_dict[stk.InchiKey().get_key(mol)] = (
                stk.BuildingBlock.init_from_molecule(
                    mol, functional_groups=[stk.BromoFactory()]
                )
            )
        self.bbs_dict = bbs_dict
        return bbs_dict

    def Build_polymers(self, element: pd.DataFrame):
        bbs_dict = self.bbs_dict

        genes = "ABCDEFGH"
        genes = genes[: self.oligomer_size]
        # print(genes)
        repeating_unit = ""
        # joins the Genes to make a repeating unit string
        repeating_unit = repeating_unit.join(genes)
        InchiKey_cols = [col for col in element.columns if "InChIKey_" in col]

        # print(element[InchiKey_cols].values.flatten())
        def gen_mol(elem):
            precursors = []
            for fragment in elem[InchiKey_cols].values.flatten():
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
            atom_types = list(
                [
                    atom.get_atom().get_atomic_number()
                    for atom in polymer.get_atom_infos()
                ]
            )
            atom_types = torch.tensor(atom_types, dtype=torch.long)

            bb_key = self.join_keys(polymer)
            molecule = Data(
                x=atom_types,
                positions=positions,
                InChIKey=stk.InchiKey().get_key(polymer),
                bb_key=bb_key,
                # y=elem['target']
            )
            return molecule

        element["polymer"] = element.swifter.apply(
            lambda x: gen_mol(x), axis=1
        )
        return element["polymer"].tolist()
