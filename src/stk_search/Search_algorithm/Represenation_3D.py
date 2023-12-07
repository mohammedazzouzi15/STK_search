
"""
this script is to encode the representation of the oligomer from the representation of the fragments
"""
import numpy as np
import torch
from torch_geometric.data import Data, Batch


class Representation3D:
    def __init__(self, model_embedding,df_results, data=None,db_poly=None,device=None):
        self.model_embedding = model_embedding
        self.model_embedding.eval()
        if device is None:
            self.device = ("cuda" if torch.cuda.is_available() else torch.device("cpu"))
        else:
            self.device = device
        self.model_embedding.to(self.device)
        self.df_inchikey = None
        self.df_total = df_results
        if data is not None:
            self.dataset = data
            if db_poly is not None:
                self.db_poly = db_poly
        elif db_poly is not None:
            self.dataset = None
            self.db_poly = db_poly
        else:
            raise ValueError("Please provide either data or db_poly")
    def generate_repr(self, elements):
        elements_copy = elements.copy()
        elements_copy = elements_copy.values
        InChIKeys = self._find_elem_InchiKey(elements_copy)
        molecules = []
        for x in InChIKeys:
            if self.dataset is not None:
                data = self.find_data(x)
                if data is not None:
                    molecules.append(data)
                else:
                    molecules.append(self._getinfo_db(x))
            else:
                molecules.append(self._getinfo_db(x))
        with torch.no_grad():
            batch = Batch.from_data_list(molecules).to(self.device)
            opt_geom_encoding = self.model_embedding(batch.x, batch.positions, batch.batch)
        return opt_geom_encoding
    def _getinfo_db(self,InChIKey):
        polymer = self.db_poly.get({"InChIKey": InChIKey[0]})
        frags = []
        dat_list = list(polymer.get_atomic_positions())
        positions = np.vstack(dat_list)
        positions = torch.tensor(positions, dtype=torch.float, device=self.device)
        atom_types = list(
            [
                atom.get_atom().get_atomic_number()
                for atom in polymer.get_atom_infos()
            ]
        )
        atom_types = torch.tensor(atom_types, dtype=torch.long, device=self.device)
        molecule = Data(x=atom_types, positions=positions, device=self.device)
        return molecule

    def find_data(self, key):
        for i in range(len(self.dataset)):
            if self.dataset[i]["InChIKey"] == key:
                data = self.dataset[i].to(self.device)
                return Data(x=data.x, positions=data.positions, device=self.device)
        print("No data found in database")
        return None
    def _find_elem_InchiKey(self, elements):
        InChIKeys = []
        if self.df_inchikey is None:
            for elm in elements:
                df_search = self.df_total.copy()
                for i, x in enumerate(elm):
                    df_search = df_search[df_search[f"InChIKey_{i}"] == x]
                InChIKeys.append(df_search['InChIKey'].values.astype(str))
        else:
            InChIKeys = self.df_inchikey['InChIKey'].values.astype(str)
        return InChIKeys
    
class Representation3DFrag:
    def __init__(self, model_embedding,model_encoding,df_results, data=None,db_poly=None,db_frag=None):
        
        self.device = ("cuda" if torch.cuda.is_available() else torch.device("cpu"))
        self.model_embedding = model_embedding
        self.model_embedding.eval()
        self.model_embedding.to(self.device)
        self.model_encoding = model_encoding
        self.model_encoding.eval()
        self.model_encoding.to(self.device)
        self.df_inchikey = None
        self.df_total = df_results
        if data is not None:
            self.dataset = data
            if db_poly is not None:
                self.db_poly = db_poly
        elif db_poly is not None:
            self.db_poly = db_poly
        else:
            raise ValueError("Please provide either data or db_poly")
        self.db_frag = db_frag
    def generate_repr(self, elements):
        elements_copy = elements.copy()
        elements_copy = elements_copy.values
        #InChIKeys = self._find_elem_InchiKey(elements_copy)
        molecules = []
        for x in elements_copy:
            if self.dataset is not None:
                InChIKeys = self._find_elem_InchiKey([x])
                data = self.find_data(InChIKeys)
                if data is not None:
                    molecules.append(data)
                else:
                    molecules.append(self._getinfo_db(x))
            else:
                molecules.append(self._getinfo_db(x))
        with torch.no_grad():
            batch = Batch.from_data_list(molecules).to(self.device)
            opt_geom_encoding = self.model_encoding(batch.x)
        return opt_geom_encoding
    def _getinfo_db(self,elements):
        frags = []
        for elm in elements:
            molecule_bb = self.db_frag.get({"InChIKey": elm})
            dat_list = list(molecule_bb.get_atomic_positions())
            positions = np.vstack(dat_list)
            positions = torch.tensor(positions, dtype=torch.float, device=self.device)
            atom_types = list(
                [atom.get_atomic_number() for atom in molecule_bb.get_atoms()]
            )
            atom_types = torch.tensor(atom_types, dtype=torch.long, device=self.device)
            molecule_frag = Data(
                x=atom_types,
                positions=positions,
                device=self.device,
            )
            frags.append(molecule_frag)
        # get the fragment based representation
        with torch.no_grad():
            self.model_embedding.eval()
            batch = Batch.from_data_list(frags).to(self.device)
            original_encoding = self.model_embedding(batch.x, batch.positions, batch.batch)
            original_encoding = original_encoding.reshape((-1,))
            original_encoding = original_encoding.unsqueeze(0)
        return Data(x=original_encoding)


    def find_data(self, key):
        for i in range(len(self.dataset)):
            if self.dataset[i]["InChIKey"] == key[0]:
                data = self.dataset[i].to(self.device)
                return Data(x=data.x)
        print("No data found in database")
        return None
    def _find_elem_InchiKey(self, elements):
        InChIKeys = []
        if self.df_inchikey is None:
            for elm in elements:
                df_search = self.df_total.copy()
                for i, x in enumerate(elm):
                    df_search = df_search[df_search[f"InChIKey_{i}"] == x]
                InChIKeys.append(df_search['InChIKey'].values.astype(str))
        else:
            InChIKeys = self.df_inchikey['InChIKey'].values.astype(str)
        return InChIKeys


class Representation3DFrag_transformer:
    def __init__(self, model_encoding,df_results, data=None,db_poly=None,db_frag=None,device=None):
        
        if device is None:
            self.device = ("cuda" if torch.cuda.is_available() else torch.device("cpu"))
        else:
            self.device = device
        self.model_encoding = model_encoding
        self.model_encoding.eval()
        self.model_encoding.to(self.device)
        self.df_inchikey = None
        self.df_total = df_results
        if data is not None:
            self.dataset = data
            if db_poly is not None:
                self.db_poly = db_poly
        elif db_poly is not None:
            self.dataset = None
            self.db_poly = db_poly
        else:
            raise ValueError("Please provide either data or db_poly")
        self.db_frag = db_frag
    def generate_repr(self, elements):
        elements_copy = elements.copy()
        elements_copy = elements_copy.values
        #InChIKeys = self._find_elem_InchiKey(elements_copy)
        molecules = []
        for x in elements_copy:
            if self.dataset is not None:
                InChIKeys = self._find_elem_InchiKey([x])
                data = self.find_data(InChIKeys)
                if data is not None:
                    molecules.append(data)
                else:
                    molecules.append(self._getinfo_db(x))
            else:
                molecules.append(self._getinfo_db(x))
                
        with torch.no_grad():
            #batch = Batch.from_data_list(molecules).to(self.device)
            opt_geom_encoding = []
            #data_loader = DataLoader(molecules, batch_size=min(100,len(molecules))) # need to check how to use batches
            #preds.view(-1, preds.size(-1))
            for batch in molecules:#data_loader:
                encoding = self.model_encoding(batch)
                opt_geom_encoding.append(encoding[0][0])

        return torch.stack(opt_geom_encoding)
    def _getinfo_db(self,elements):
        frags = []
        for elm in elements:
            molecule_bb = self.db_frag.get({"InChIKey": elm})
            dat_list = list(molecule_bb.get_atomic_positions())
            positions = np.vstack(dat_list)
            positions = torch.tensor(positions, dtype=torch.float, device=self.device)
            atom_types = list(
                [atom.get_atomic_number() for atom in molecule_bb.get_atoms()]
            )
            atom_types = torch.tensor(atom_types, dtype=torch.long, device=self.device)

            molecule_frag = Data(
                x=atom_types,
                positions=positions,
                device=self.device,
            )
            frags.append(molecule_frag)
        return frags


    def find_data(self, key):
        for i in range(len(self.dataset)):
            if self.dataset[i][0]["InChIKey"] == key[0]:
                data = self.dataset[i].to(self.device)
                return Data(x=data.x)
        print("No data found in database")
        return None
    def _find_elem_InchiKey(self, elements):
        InChIKeys = []
        if self.df_inchikey is None:
            for elm in elements:
                df_search = self.df_total.copy()
                for i, x in enumerate(elm):
                    df_search = df_search[df_search[f"InChIKey_{i}"] == x]
                InChIKeys.append(df_search['InChIKey'].values.astype(str))
        else:
            InChIKeys = self.df_inchikey['InChIKey'].values.astype(str)
        return InChIKeys

