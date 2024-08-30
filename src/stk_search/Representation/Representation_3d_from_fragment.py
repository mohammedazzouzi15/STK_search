"""Encode the Representation of the oligomer from the Representation of the fragments using a pretrained GNN.

here we use a model to encode the Representation of the fragments and then use this Representation to encode the Representation of the oligomer.
"""

import numpy as np
import pymongo
import torch
from pymongo import UpdateOne
from torch_geometric.data import Data


class Representation3dFromFragment:
    """Generate the Representation of the constructed molecules from the fragment representation.

    In this class we use the 3d Representation of the fragements directly as input to the model that will generate the Representation of the oligomer.
    the model has been trained to encode the Representation of the fragments and then use this Representation to encode the Representation of the oligomer.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_encoding,
        df_results,
        data=None,
        db_poly=None,
        db_frag=None,
        device=None,
    ):
        """Initialise the class.

        Args:
        ----
            model_encoding (torch.nn.Module): model used to encode the Representation of the fragments
            df_results (pd.dataframe): table of building blocks nmaed with their InChIKey
            data (list): list of data containing the Representation of the fragments
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
        self.model_encoding = model_encoding
        self.model_encoding.eval()
        self.model_encoding.to(self.device)
        self.dataset_local = {}
        self.dataset_frag_local = {}
        self.df_inchikey = None
        self.df_total = df_results
        if data is not None:
            self.dataset = data.copy()
            if db_poly is not None:
                self.db_poly = db_poly
        elif db_poly is not None:
            self.dataset = None
            self.db_poly = db_poly
        else:
            msg = "Please provide either data or db_poly"
            raise ValueError(msg)
        self.db_frag = db_frag
        self.name = "Representation3dFromFragment"
        self.save_dataset_path = ""
        self.db_name = "test"
        self.mongodb_client = pymongo.MongoClient("mongodb://localhost:27017/")

    def generate_repr(self, elements):
        """Generate the Representation of the elements.

        Args:
        ----
            elements (pd.dataframe): table of building blocks nmaed with their InChIKey
        Returns:
            torch.tensor: Representation of the constructed molecule

        """
        elements_copy = elements.copy()
        opt_geom_encoding = []
        if self.dataset is not None:
            # Create a dictionary that maps InChIKeys to data
            dataset_dict = {x.InChIKey: x for x in self.dataset}
            InChIKeys = self._find_elem_InchiKey(elements_copy)  # noqa: N806
            for _id, InChIKey in enumerate(InChIKeys):  # noqa: N806
                data = dataset_dict.get(InChIKey)  # self.find_data(InChIKey)
                if data is not None:
                    opt_geom_encoding.append(data.learned_rpr)
                else:
                    molecule, key = self._getinfo_db(
                        elements_copy.to_numpy()[_id]
                    )
                    with torch.no_grad():
                        encoding = self.model_encoding(molecule)
                        opt_geom_encoding.append(encoding[0][0])
                        self.dataset.append(
                            Data(
                                InChIKey=InChIKey,
                                learned_rpr=encoding[0][0].type(torch.float16),
                            )
                            .detach()
                            .cpu()
                        )
        else:
            dataset_local_new = {}
            for x in elements_copy.to_numpy():
                key = ""
                for elm in x:
                    key = elm + key
                if key in self.dataset_local:
                    opt_geom_encoding.append(self.dataset_local[key])
                else:
                    molecule, key = self._getinfo_db(x)
                    with torch.no_grad():
                        encoding = self.model_encoding(molecule)
                        opt_geom_encoding.append(encoding[0][0])
                        self.dataset_local[key] = (
                            encoding[0][0].type(torch.float16).detach()
                        )
                        dataset_local_new[key] = (
                            encoding[0][0].type(torch.float16).detach()
                        )
        return torch.stack(opt_geom_encoding)

    def _getinfo_db(self, elements) -> tuple:
        """Get the information from the database.

        Args:
        ----
            elements (list): list of InChIKeys

        Returns:
        -------
            frags: list of data containing the Representation of the fragments
            key: key to save the Representation of the fragments in the local dataset

        """
        frags = []
        key = ""
        for elm in elements:
            if elm in self.dataset_frag_local:
                frags.append(self.dataset_frag_local[elm])
                key = elm + key
                continue
            molecule_bb = self.db_frag.get({"InChIKey": elm})
            dat_list = list(molecule_bb.get_atomic_positions())
            positions = np.vstack(dat_list)
            positions = torch.tensor(
                positions, dtype=torch.float, device=self.device
            )
            atom_types = [
                atom.get_atomic_number() for atom in molecule_bb.get_atoms()
            ]
            atom_types = torch.tensor(
                atom_types, dtype=torch.long, device=self.device
            )

            molecule_frag = Data(
                x=atom_types,
                positions=positions,
                device=self.device,
            )
            frags.append(molecule_frag)
            self.dataset_frag_local[elm] = molecule_frag
            key = elm + key
        return frags, key

    def _find_elem_InchiKey(self, elements) -> list:
        """Find the InChIKey of the elements.

        Args:
        ----
            elements (pd.dataframe): table of building blocks nmaed with their InChIKey
        Returns:
            list: list of InChIKeys

        """
        num_fragment = elements.shape[1]
        results = elements.merge(
            self.df_total,
            on=[f"InChIKey_{i}" for i in range(num_fragment)],
            how="left",
        )
        results = results.drop_duplicates(
            subset=[f"InChIKey_{i}" for i in range(num_fragment)]
        )
        if results.shape[0] != elements.shape[0]:
            msg = "InChIKey not found in database"
            raise ValueError(msg)

        return results["InChIKey"].tonumpy()

    def save_dataset_local(self):
        """Save the dataset_local."""
        if self.save_dataset_path == "":
            pass
        else:
            torch.save(self.dataset_local, self.save_dataset_path)

    def add_representation_to_local_dataset(self, elements):
        """Add the Representation to the local dataset.

        Args:
        ----
            elements (pd.dataframe): table of building blocks nmaed with their InChIKey
            Returns:
            dict: dictionary containing the Representation of the elements

        """
        df_element = elements.copy()
        client = pymongo.MongoClient("mongodb://ch-atarzia.ch.ic.ac.uk/")
        db = client["learned_representations"]
        collection = db[self.db_name]
        df_element["key"] = df_element.apply(lambda x: "".join(x), axis=1)
        keys = df_element["key"].to_list()
        keys = [x for x in keys if x not in self.dataset_local]
        for document in collection.find({"key": {"$in": keys}}):
            self.dataset_local[document["key"]] = torch.tensor(
                document["representation"]
            )
        return self.dataset_local

    def save_representation_to_database(self, local_dataset_new):
        """Save the Representation to the database.
        
        Args:
        ----
            local_dataset_new (dict): dictionary containing the Representation of the elements

        Returns:
        -------
            dict: dictionary containing the Representation of the elements

        """
        client = pymongo.MongoClient(self.mongodb_client)
        db = client["learned_representations"]
        collection = db[self.db_name]
        if len(local_dataset_new) == 0:
            return local_dataset_new
        collection.bulk_write(
            [
                UpdateOne(
                    {"key": key},
                    {
                        "$set": {
                            "representation": local_dataset_new[key].tolist()
                        }
                    },
                    upsert=True,
                )
                for key in local_dataset_new
            ]
        )
        return local_dataset_new
