"""Lis of functions for the generation of precursors and the generation of precursors database.

The functions in this module are used to generate precursors and the database of precursors.
"""

import itertools
from pathlib import Path

import pandas as pd
import pymongo
import stk
import stko
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from stk_search.ObjectiveFunctions.IpEs1Fosc import IpEs1Fosc


# helper function to define building blocks from fragments
def fragment_from_smiles(smile):
    """Make fragments from SMILES string.

    This function takes a SMILES string as the initial molecular fragment and returns a list of RDKit molecules for the building blocks.
    Here we place a bromin atoms to identify the potential connection points.
    the building of an oligomer would use the stk.bromoFactory to connect the building blocks.
     https://stk.readthedocs.io/en/stable/_autosummary/stk.BromoFactory.html#

    Args:
    ----
    smile : str
        SMILES string

    Returns:
    -------
    mol_list : list
        list of RDKit molecules
    mol_list_smiles : list
        list of SMILES strings

    """
    mol2 = Chem.MolFromSmiles(smile)
    mol2 = Chem.AddHs(mol2)
    if rdMolDescriptors.CalcNumAromaticRings(mol2) > 6:
        return [], [], []
    if mol2.GetNumAtoms() > 40:
        return [], [], []
    potential_connection = []
    for x in mol2.GetAtoms():
        if (
            x.GetAtomicNum() == 35
            or x.GetAtomicNum() == 34
            or x.GetAtomicNum() == 14
            or x.GetAtomicNum() == 17
        ):
            return [], [], []
        if x.GetAtomicNum() == 6 and x.GetHybridization().name == "SP2":
            potential_connection.extend(
                [
                    atom.GetIdx()
                    for atom in x.GetNeighbors()
                    if atom.GetAtomicNum() == 1
                ]
            )
    mol_list = []
    mol_list_smiles = []
    potential_connection = list(set(potential_connection))
    for atom1, atom2 in set(itertools.combinations(potential_connection, 2)):
        mol_trans = Chem.Mol(mol2)
        pass_case = False
        neighbour_atom1 = mol_trans.GetAtoms()[atom1].GetNeighbors()
        neighbour_atom2 = mol_trans.GetAtoms()[atom2].GetNeighbors()
        for atom1_n in neighbour_atom1:
            for atom2_n in neighbour_atom2:
                if atom1_n.GetIdx() == atom2_n.GetIdx():
                    pass_case = True
                    break
                if atom1_n.GetIdx() in [
                    x.GetIdx() for x in atom2_n.GetNeighbors()
                ]:
                    pass_case = True
                    break
                if (
                    len(
                        set(
                            [x.GetIdx() for x in atom1_n.GetNeighbors()]
                        ).intersection(
                            set([x.GetIdx() for x in atom2_n.GetNeighbors()])
                        )
                    )
                    > 0
                ):
                    pass_case = True
                    break
        if pass_case:
            continue
        mol_trans.GetAtoms()[atom1].SetAtomicNum(35)
        mol_trans.GetAtoms()[atom2].SetAtomicNum(35)
        mol_list.append(mol_trans)
        mol_list_smiles.append(Chem.MolToSmiles(mol_trans))
    return mol_list, mol_list_smiles, [smile]


def load_data(
    collection_name="Precursors",
    client="mongodb://localhost:27017/",
    database="stk_mohammed_precursor",
    excited_state_calculated=False,
    xtb_calculated=False,
):
    """Load data from the database.

    This function loads the data from the database and returns a pandas dataframe.
    Here we consider that the database contains the molecules, the excited state properties and the xtb properties.

    Args:
    ----
    collection_name : str
        The name of the collection in the database.
    client : str
        The client for the database.
    database : str
        The name of the database.
    excited_state_calculated : bool
        If the excited states are calculated.
    xtb_calculated : bool
        If the xtb properties are calculated.

    Returns:
    -------
        df_total : pd.DataFrame
            The dataframe containing the data from the database.

    """
    client = pymongo.MongoClient(client)
    database = client[database]

    collection = database["molecules"]
    df_total = pd.DataFrame(list(collection.find()))
    if xtb_calculated:
        collection = database[f"{collection_name}_IPEA"]
        df_IPEA = pd.DataFrame(list(collection.find()))
        collection = database[f"{collection_name}_opt"]
        df_opt = pd.DataFrame(list(collection.find()))
        df_total = df_total.merge(df_IPEA, on="InChIKey", how="inner")
        df_opt.drop(columns=["_id", "total energy (au)"], inplace=True)
        df_total = df_total.merge(df_opt, on="InChIKey", how="outer")
    if excited_state_calculated:
        collection = database[f"{collection_name}_Stda"]
        df_STDA = pd.DataFrame(list(collection.find()))
        df_total = df_total.merge(df_STDA, on="InChIKey", how="inner")
        df_total.dropna(subset=["Excited state energy (eV)"], inplace=True)
        df_total["ES1"] = df_total["Excited state energy (eV)"].apply(
            lambda x: x[0]
        )
        df_total["fosc1"] = df_total[
            "Excited state oscillator strength"
        ].apply(lambda x: x[0])
    return df_total


class PrecursorDatabase(IpEs1Fosc):
    """A class to add the precursors to the database."""

    def __init__(
        self,
        client_address="mongodb://localhost:27017/",
        database_name="stk__precursor",
    ):
        """Initialise the class."""
        self.client_address = client_address
        self.client = pymongo.MongoClient(client_address)
        self.database_name = database_name
        self.database = self.client[database_name]
        self.collection = self.database["molecules"]
        self.collection_name = "Precursors"

    def add_precursor_to_database(
        self, smile, functional_groups=stk.BromoFactory
    ):
        """Add a precursor to the database.

        This function adds the precursors to the database.

        Args:
        ----
        smile : str
            The SMILES string of the precursor.
        functional_groups : stk.FunctionalGroup
            The functional groups of the precursor.

        Returns:
        -------
            stk_mol : stk.BuildingBlock
                The building block of the precursor.

        """
        precursor_database = stk.MoleculeMongoDb(
            self.client,
            database=self.database_name,
        )
        stk_mol = stk.BuildingBlock(
            smile, functional_groups=[functional_groups()]
        )
        precursor_database.put(stk_mol)
        return stk_mol

    def build_polymer(self, smile) -> stk.BuildingBlock:
        """Build the molecule from the smile."""
        return stk.BuildingBlock(smile, functional_groups=[stk.BromoFactory()])

    def initialise_connections(self):
        """Initialise the connections to the database and the quantum chemical software."""
        self.test_mongo_db_connection()
        self.test_xtb_stda_connection()
        self.db_polymer = stk.MoleculeMongoDb(
            self.client,
            database=self.database_name,
        )

    def load_data(self):
        """Load data from the database.

        This function loads the data from the database and returns a pandas dataframe.
        Here we consider that the database contains the molecules, the excited state properties and the xtb properties.


        Returns
        -------
            df_total : pd.DataFrame
                The dataframe containing the data from the database.

        """

        def add_data_to_df_total(collection, df_total) -> pd.DataFrame:
            """Add the data to the dataframe."""
            collection = self.database[collection]
            df_toadd = pd.DataFrame(list(collection.find()))
            df_toadd.drop(columns=["_id"], inplace=True)
            return df_total.merge(df_toadd, on="InChIKey", how="inner")

        database = self.database
        collection = database["molecules"]
        df_total = pd.DataFrame(list(collection.find()))
        for collections in self.extra_collections:
            df_total = add_data_to_df_total(collections, df_total)
        return df_total


def get_inchi_key(molecule) -> str:
    """Get the inchi key of the molecule."""
    return stk.InchiKey().get_key(molecule)


class CalculatePrecursorIPEs1Fosc(IpEs1Fosc):
    """A class to calculate the excited state properties using sTDA method from xtb output.

    Attributes
    ----------
    oligomer_size : int
        The size of the oligomer.
    database_output_folder : str
        The path to the database folder.
    collection_name : str
        The name of the collection.


    """

    def __init__(self, oligomer_size, database_output_folder):
        """Initialise the class."""
        super().__init__(oligomer_size, database_output_folder)
        self.collection_name = "stk_precursor"

    def initialise_connections(self):
        """Initialise the connections to the database and the quantum chemical software."""
        self.test_mongo_db_connection()
        self.test_xtb_stda_connection()
        self.db_polymer = stk.MoleculeMongoDb(
            self.client,
            database=self.database_name,
        )

    def build_polymer(self, smile) -> stk.BuildingBlock:
        """Build the molecule from the smile."""
        return stk.BuildingBlock(smile, functional_groups=[stk.BromoFactory()])

    def run_ETKDG_opt(
        self,
        smile,
        xtb_opt_output_dir,
        database="stk_mohammed_BO",
        client=None,
    ):
        precursor = self.build_polymer(smile)
        Path(xtb_opt_output_dir, get_inchi_key(precursor))
        etkdg = stko.OptimizerSequence(
            stko.ETKDG(),
        )
        precursor = etkdg.optimize(precursor)
        db_polymer = stk.MoleculeMongoDb(
            client,
            database=database,
        )
        db_polymer.put(precursor)
        return precursor
