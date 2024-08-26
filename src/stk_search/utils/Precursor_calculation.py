"""A module for calculating the precursor molecule using stk and stko.
Here the input is the smile of the molecule and the output is the precursor molecule."""

import os
import re
from pathlib import Path

import pymongo
import stk
import stko

from stk_search.ObjectiveFunction import IpEs1Fosc


def get_inchi_key(molecule)->str:
    """Get the inchi key of the molecule"""
    return stk.InchiKey().get_key(molecule)


class CalculatePrecursor(IpEs1Fosc):
    """A class to calculate the excited state properties using sTDA method from xtb output.
    
    Attributes
    ----------
    oligomer_size : int
        The size of the oligomer.
    db_folder : str
        The path to the database folder.
    collection_name : str
        The name of the collection.
    

    """

    def __init__(self, oligomer_size, db_folder):
        super().__init__(oligomer_size, db_folder)
        self.collection_name = "stk_precursor"

    def initialise_connections(self):
        """Initialise the connections to the database and the quantum chemical software."""
        self.test_mongo_db_connection()
        self.test_xtb_stda_connection()
        self.db_polymer = stk.MoleculeMongoDb(
                self.client,
                database=self.database_new_calc,
            )
        
    def build_polymer(self, smile)->stk.BuildingBlock:
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
