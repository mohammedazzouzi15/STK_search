"""Module for the IpEs1Fosc objective function.

The IpEs1Fosc objective function is used to evaluate the fitness of the molecules by calculating the ionisation potential,
the first excited state energy and the first excited state oscillator strength.
The fitness function is defined as:
-np.abs(IP - 5.5) - 0.5 * np.abs(Es1 - 3) + np.log10
where IP is the ionisation potential, Es1 is the first excited state energy and fosc_1 is the first excited state oscillator strength
Here the quantum chemical calculation are done using xtb and stda.
"""

import logging
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pymongo
import stk
import stko

from stk_search.Calculators.STDA_calculator import sTDAXTB
from stk_search.Calculators.XTBcalculator import XTBEnergy2
from stk_search.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


def get_inchi_key(molecule):
    """Get the InChIKey of the molecule.

    Args:
    ----
        molecule: stk.BuildingBlock
        the molecule

    Returns:
    -------
        str
        the InChIKey of the molecule

    """
    return stk.InchiKey().get_key(molecule)

class IpEs1Fosc(ObjectiveFunction):
    """Class for the IpEs1Fosc objective function.

    The IpEs1Fosc objective function is used to evaluate the fitness of the molecules by calculating the ionisation potential,
    the first excited state energy and the first excited state oscillator strength.
    The fitness function is defined as:
    -np.abs(IP - 5.5) - 0.5 * np.abs(Es1 - 3) + np.log10
    where IP is the ionisation potential, Es1 is the first excited state energy and fosc_1 is the first excited state oscillator strength
    Here the quantum chemical calculation are done using xtb and stda.

    Properties
    ----------
    client: str
    the path to the mongodb client
    database_name: str
    the name of the database containing the building blocks
    the database should contain the building blocks position matrix and the InChIKey
    It is normally generated using stk and the stk.MoleculeMongoDb class
    xtb_path: str
    the path to the xtb executable
    stda_bin_path: str
    the path to the stda executable
    database_output_folder: str
    the path to the output directory
    database_name: str
    the name of the database containing the new calculations
    collection_name: str
    the name of the collection
    host_ip: str
    the host IP
    oligomer_size: int
    the size of the oligomer

    Functions
    ---------
    evaluate_element(element)
        Evaluates the fitness of the element
        takes as an input a list of building blocks and returns the fitness of the element

    Build_polymer(element, db)
        Builds the polymer from the building blocks
        takes as an input a list of building blocks and a database containing the building blocks
        returns the polymer
    run_xtb_opt(polymer, xtb_path, xtb_opt_output_dir, database, collection, client)
        Runs the xtb optimisation of the polymer
        takes as an input the polymer, the path to xtb, the output directory, the database and collection name and the client
        returns the optimised polymer
    run_xtb_ipea(polymer, xtb_path, xtb_opt_output_dir, database, collection, target, client)
        Runs the xtb calculation of the ionisation potential
        takes as an input the polymer, the path to xtb, the output directory, the database and collection name, the target and the client
        returns the ionisation potential
    run_stda(polymer, stda_bin_path, output_dir, property, state, database, collection, client)
        Runs the stda calculation of the excited state energy and oscillator strength
        takes as an input the polymer, the path to stda, the output directory, the property, the state, the database and collection name and the client
        returns the excited state energy or oscillator strength



    """

    def __init__(
        self,
        oligomer_size,
        database_output_folder="/rds/general/ephemeral/user/ma11115/ephemeral/BO_polymers",
    ):
        """Initialise the IpEs1Fosc objective function.

        Args:
        ----
            oligomer_size: int
            the size of the oligomer
            database_output_folder: str
            the path to the output directory
            database_name: str
            the name of the database containing the new calculations
            collection_name: str
            the name of the collection
            host_ip: str
            the host IP

        """
        super().__init__()
        self.client_address = "mongodb://localhost:27017/"
        self.database_name = "stk_constructed"
        self.xtb_path = "xtb"
        self.stda_bin_path = "stda"
        self.database_output_folder = Path(database_output_folder)
        Path.mkdir(self.database_output_folder, exist_ok=True)
        self.database_name = "stk_constructed"
        self.collection_name = f"BO_{oligomer_size}"
        self.host_ip = "localhost"
        self.oligomer_size = oligomer_size
        self.db_polymer = None
        self.client = None

    def initialise_connections(self):
        """Initialise the connections to the database and the quantum chemical software."""
        self.test_mongo_db_connection()
        self.test_xtb_stda_connection()
        self.db_polymer = stk.ConstructedMoleculeMongoDb(
            self.client,
            database=self.database_name,
        )

    def test_mongo_db_connection(self):
        """Tests the connection to the database."""
        try:
            self.client = pymongo.MongoClient(self.client_address)
            stk.MoleculeMongoDb(
                self.client,
                database=self.database_name,
            )
            logging.info("Connected to the database")
        except pymongo.errors.ConfigurationError:
            logging.exception("Error connecting to the database: %s")

    def test_xtb_stda_connection(self):
        """Tests the connection to xtb and stda."""
        try:
            command = self.xtb_path + " --version"
            subprocess.run(command, check=True, shell=True)  # noqa: S602
            logging.info("Connected to xtb")
        except subprocess.CalledProcessError:
            logging.exception("Error connecting to xtb %s")
        try:
            command = self.stda_bin_path + "/stda_v1.6.1 --version"
            subprocess.run(command, check=True, shell=True)  # noqa: S602
            logging.info("Connected to stda")
        except subprocess.CalledProcessError:
            logging.exception("Error connecting to stda %s")

    def evaluate_element(self, element):
        """Evaluate the fitness of the element.

        Takes as an input a list of building blocks and returns the fitness of the element
        The evaluation here is done by first building the polymer from the building blocks
        then running the xtb optimisation, the xtb calculation of the ionisation potential and the stda calculation of the excited state energy and oscillator strength
        The fitness function is defined as:
        -np.abs(IP - 5.5) - 0.5 * np.abs(Es1 - 3) + np.log10(fosc_1 + 1e-10)
        where IP is the ionisation potential, Es1 is the first excited state energy and fosc_1 is the first excited state oscillator strength.

        Args:
        ----
            element: list
            list of building blocks

        Returns:
        -------
            float
            the fitness of the element
            str
            the identifier of the element in the form of an InChIKey

        """
        # initialise the database

        # define the output directories
        self.initialise_connections()
        output_dir_ipea = Path(self.database_output_folder, "Database", "xtb_calculations")
        xtb_opt_output_dir = Path(
            self.database_output_folder, "Database", "xtb_opt_output_dir"
        )
        output_dir_stda = Path(self.database_output_folder, "Database", "stda_output_dir")
        Path.mkdir(output_dir_ipea, exist_ok=True, parents=True)
        Path.mkdir(xtb_opt_output_dir, exist_ok=True, parents=True)
        Path.mkdir(output_dir_stda, exist_ok=True, parents=True)
        # build the polymer
        polymer = self.build_polymer(element)
        polymer = self.run_xtb_opt(
            polymer,
            self.xtb_path,
            xtb_opt_output_dir,
        )
        Inchikey = stk.InchiKey().get_key(polymer)  # noqa: N806
        ip = self.run_xtb_ipea(
            polymer,
            self.xtb_path,
            output_dir_ipea,
            target="ionisation potential (eV)",
        )
        es1 = self.run_stda(
            polymer,
            self.stda_bin_path,
            output_dir_stda,
            excited_state_property="Excited state energy (eV)",
            state=0,
        )
        fosc_1 = self.run_stda(
            polymer,
            self.stda_bin_path,
            output_dir_stda,
            excited_state_property="Excited state oscillator strength",
            state=0,
        )
        fitness_function = (
            -np.abs(ip - 5.5)
            - 0.5 * np.abs(es1 - 3)
            + np.log10(fosc_1 + 1e-10)
        )
        self.client.close()
        return fitness_function, Inchikey

    def build_polymer(self, element: pd.DataFrame):
        """Build the polymer from the building blocks.

        Takes as an input a list of building blocks and a database containing the building blocks
        returns the polymer.

        Args:
        ----
            element: pd.DataFrame
            the dataframe containing the building blocks
            db: stk.MoleculeMongoDb
            the database containing the building blocks

        Returns:
        -------
            stk.ConstructedMolecule
            the polymer

        """
        database_name = stk.MoleculeMongoDb(
            self.client,
            database=self.database_name,
        )
        precursors = []
        genes = "ABCDEFGH"
        genes = genes[: self.oligomer_size]
        repeating_unit = ""
        # joins the Genes to make a repeating unit string
        repeating_unit = repeating_unit.join(genes)
        InchiKey_cols = [col for col in element.columns if "InChIKey_" in col]  # noqa: N806
        for fragment in element[InchiKey_cols].to_numpy().flatten():
            mol = database_name.get({"InChIKey": fragment})
            bb = stk.BuildingBlock.init_from_molecule(
                mol, functional_groups=[stk.BromoFactory()]
            )
            precursors.append(bb)
        return stk.ConstructedMolecule(
            stk.polymer.Linear(
                building_blocks=precursors,
                repeating_unit=repeating_unit,
                num_repeating_units=1,
            )
        )

    def run_xtb_opt(
        self,
        polymer,
        xtb_path,
        xtb_opt_output_dir,
    ):
        """Run the xtb optimisation of the polymer.

        Args:
        ----
            polymer: stk.ConstructedMolecule
            the polymer
            xtb_path: str
            the path to the xtb executable
            xtb_opt_output_dir: str
            the output directory
            database: str
            the name of the database
            collection: str
            the name of the collection
            client: pymongo.MongoClient
            the client

        Returns:
        -------
            stk.ConstructedMolecule
            the optimised polymer

        """

        def save_xtb_opt_calculation(
            polymer,
            xtb_opt_output_dir,
            collection=None,
            InchiKey_initial=None,  # noqa: N803
        ) -> None:
            """Save the xtb optimisation calculation.

            Args:
            ----
                polymer: stk.ConstructedMolecule
                the polymer
                xtb_opt_output_dir: str
                the output directory
                collection: pymongo.collection
                the collection
                InchiKey_initial: str
                the initial InChIKey


            Returns:
            -------
            None

            """

            def get_property_value(data, property_name) -> float:
                for line in data:
                    if property_name in line:
                        if property_name == "cpu-time":
                            return (
                                re.findall(r"[-+]?(?:\d*\.*\d+)", line)[-3]
                                + " h "
                                + re.findall(r"[-+]?(?:\d*\.*\d+)", line)[-2]
                                + " min "
                                + re.findall(r"[-+]?(?:\d*\.*\d+)", line)[-1]
                                + " s "
                            )
                        return float(
                            re.findall(r"[-+]?(?:\d*\.*\d+)", line)[-1]
                        )  # float(words[3]) #
                return None

            polymer_xtb_opt_calc = {
                "InChIKey": stk.InchiKey().get_key(polymer),
                "cal_folder": str(
                    Path(
                        xtb_opt_output_dir, stk.InchiKey().get_key(polymer)
                    ).absolute()
                ),
                "Host IP": self.host_ip,
                "InChIKey_initial": InchiKey_initial,
            }
            with Path(
                polymer_xtb_opt_calc["cal_folder"], "optimization_1.output"
            ).open(encoding="utf-8") as f:
                data = f.readlines()
            polymer_xtb_opt_calc["cpu time"] = get_property_value(
                data, "cpu-time"
            )
            polymer_xtb_opt_calc["total energy (au)"] = get_property_value(
                data, "TOTAL ENERGY"
            )
            polymer_xtb_opt_calc["HOMO-LUMO GAP (eV)"] = get_property_value(
                data, "HOMO-LUMO GAP"
            )
            collection.update_many(
                filter={"InChIKey": get_inchi_key(polymer)},
                update={"$set": polymer_xtb_opt_calc},
                upsert=True,
            )

        collection = self.client[self.database_name][
            self.collection_name + "_opt"
        ]
        if (
            collection.find_one({"InChIKey": get_inchi_key(polymer)})
            is not None
        ):
            return self.db_polymer.get({"InChIKey": get_inchi_key(polymer)})
        if (
            collection.find_one({"InChIKey_initial": get_inchi_key(polymer)})
            is not None
        ):
            data = collection.find_one(
                {"InChIKey_initial": get_inchi_key(polymer)}
            )
            return self.db_polymer.get({"InChIKey": data["InChIKey"]})
        output_dir = Path(xtb_opt_output_dir, get_inchi_key(polymer))
        InchiKey_initial = get_inchi_key(polymer)  # noqa: N806
        xtb = stko.OptimizerSequence(
            stko.ETKDG(),
            stko.XTB(
                xtb_path=xtb_path,
                output_dir=output_dir,
                unlimited_memory=False,
                num_cores=25,
            ),
        )
        polymer = xtb.optimize(polymer)
        new_output_dir = Path(xtb_opt_output_dir, get_inchi_key(polymer))
        output_dir.rename(new_output_dir)
        save_xtb_opt_calculation(
            polymer,
            xtb_opt_output_dir,
            collection=collection,
            InchiKey_initial=InchiKey_initial,
        )
        self.db_polymer.put(polymer)
        return polymer

    def run_xtb_ipea(
        self,
        polymer,
        xtb_path,
        xtb_opt_output_dir,
        target="ionisation potential (eV)",
    ):
        """Run the xtb calculation of the ionisation potential.

        Args:
        ----
            polymer: stk.ConstructedMolecule
            the polymer
            xtb_path: str
            the path to the xtb executable
            xtb_opt_output_dir: str
            the output directory
            target: str
            the target

        Returns:
        -------
            float
            the ionisation potential

        """
        collection = self.client[self.database_name][
            self.collection_name + "_IPEA"
        ]
        xtb_results = collection.find_one({"InChIKey": get_inchi_key(polymer)})
        if xtb_results is not None:
            return xtb_results[target]
        xtb = XTBEnergy2(
            xtb_path=xtb_path,
            output_dir=Path(xtb_opt_output_dir, get_inchi_key(polymer)),
            unlimited_memory=False,
            calculate_ip_and_ea=True,
            num_cores=25,
        )
        xtb_results = xtb.get_results(polymer)
        xtb_results = {
            "total energy (au)": xtb_results.get_total_energy()[0],
            "homo lumo_gap (eV)": xtb_results.get_homo_lumo_gap()[0],
            "electron affinity (eV)": xtb_results.get_electron_affinity()[0],
            "ionisation potential (eV)": xtb_results.get_ionisation_potential()[
                0
            ],
            "InChIKey": get_inchi_key(polymer),
            "cal_folder": str(
                Path(xtb_opt_output_dir, get_inchi_key(polymer)).absolute()
            ),
            "Host IP": self.host_ip,
        }
        collection.update_many(
            filter={"InChIKey": get_inchi_key(polymer)},
            update={"$set": xtb_results},
            upsert=True,
        )
        return xtb_results[target]

    def run_stda(
        self,
        polymer,
        stda_bin_path,
        output_dir,
        excited_state_property="Excited state energy (eV)",
        state=1,
    ):
        """Run XTB-stda.

        Args:
        ----
            polymer: stk.ConstructedMolecule
            the polymer
            stda_bin_path: str
            the path to the STDA xtb executable
            output_dir: str
            the output directory
            excited_state_property: str
            the property of the excited state to output
            state: int
            the excited state for which we output the property

        Returns:
        -------
            float
            The property of interrest

        """
        collection = self.client[self.database_name][
            self.collection_name + "_Stda"
        ]
        stda_results = collection.find_one(
            {"InChIKey": get_inchi_key(polymer)}
        )
        if stda_results is not None:
            return stda_results[excited_state_property][state]
        try:
            stda = sTDAXTB(
                stda_bin_path=stda_bin_path,
                num_threads=25,
                output_dir=Path(output_dir, get_inchi_key(polymer)),
            )
            excited_state_energy, excited_state_osc = stda.get_results(polymer)
            if len(excited_state_energy) > 0:
                stda_results = {
                    "Excited state energy (eV)": excited_state_energy,
                    "Excited state oscillator strength": excited_state_osc,
                    "InChIKey": get_inchi_key(polymer),
                    "cal_folder": str(
                        Path(output_dir, get_inchi_key(polymer)).absolute()
                    ),
                    "Host IP": self.host_ip,
                }
                collection.update_many(
                    filter={"InChIKey": get_inchi_key(polymer)},
                    update={"$set": stda_results},
                    upsert=True,
                )
                return stda_results[excited_state_property][state]

        except subprocess.CalledProcessError:
            logging.exception("Error running stda %s")
            return None
