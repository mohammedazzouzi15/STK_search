"""
this script is to encode the representation of the oligomer from the representation of the fragments
"""

import numpy as np
import torch
from torch_geometric.data import Data, Batch


class Representation_3d_from_fragment:
    def __init__(
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
            model_encoding (torch.nn.Module): model used to encode the representation of the fragments
            df_results (pd.dataframe): table of building blocks nmaed with their InChIKey
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
        self.model_encoding = model_encoding
        self.model_encoding.eval()
        self.model_encoding.to(self.device)
        self.dataset_local = {}
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
            raise ValueError("Please provide either data or db_poly")
        self.db_frag = db_frag
        self.name = 'Representation_3d_from_fragment'

    def generate_repr(self, elements):
        """Generate the representation of the elements.
        Args:
            elements (pd.dataframe): table of building blocks nmaed with their InChIKey
        Returns:
            torch.tensor: representation of the constructed molecule
        """

        elements_copy = elements.copy()
        opt_geom_encoding = []
        if self.dataset is not None:
            # Create a dictionary that maps InChIKeys to data
            dataset_dict = {x.InChIKey: x for x in self.dataset}
            InChIKeys = self._find_elem_InchiKey(elements_copy)
            for id, InChIKey in enumerate(InChIKeys):
                data = dataset_dict.get(InChIKey)  # self.find_data(InChIKey)
                if data is not None:
                    opt_geom_encoding.append(data.learned_rpr)
                else:
                    molecule, key = self._getinfo_db(elements_copy.values[id])
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
            for x in elements_copy.values:
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
        return torch.stack(opt_geom_encoding)

    def _getinfo_db(self, elements):
        """Get the information from the database.
        Args:
            elements (list): list of InChIKeys
        Returns:
            list: list of data containing the representation of the fragments
            str: InChIKey of the molecule
        """
        frags = []
        key = ""
        for elm in elements:
            molecule_bb = self.db_frag.get({"InChIKey": elm})
            dat_list = list(molecule_bb.get_atomic_positions())
            positions = np.vstack(dat_list)
            positions = torch.tensor(
                positions, dtype=torch.float, device=self.device
            )
            atom_types = list(
                [atom.get_atomic_number() for atom in molecule_bb.get_atoms()]
            )
            atom_types = torch.tensor(
                atom_types, dtype=torch.long, device=self.device
            )

            molecule_frag = Data(
                x=atom_types,
                positions=positions,
                device=self.device,
            )
            frags.append(molecule_frag)
            key = elm + key
        return frags, key

    def _find_elem_InchiKey(self, elements):
        """Find the InChIKey of the elements.
        Args:
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
        results.drop_duplicates(
            subset=[f"InChIKey_{i}" for i in range(num_fragment)], inplace=True
        )
        if results.shape[0] != elements.shape[0]:
            raise ValueError("InChIKey not found in database")

        return results["InChIKey"].values
