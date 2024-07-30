import os
import re

import numpy as np
import pandas as pd
import pymongo
import stk
import stko

from stk_search.Calculators.STDA_calculator import sTDA_XTB
from stk_search.Calculators.XTBcalculator import XTBEnergy2


def get_inchi_key(molecule):
    return stk.InchiKey().get_key(molecule)


class Objective_Function:
    """Base class for objective functions
    The objective function is the function that will be used to evaluate the fitness of the molecules in the search.

    Functions
    ---------
    evaluate_element(element, multiFidelity=False)
        Evaluates the fitness of the element
        takes as an input a list of building blocks and returns the fitness of the element

    """

    def __init__(self):
        """Initialises the objective function."""

    def evaluate_element(self, element, multiFidelity=False):
        """Evaluates the fitness of the element
        takes as an input a list of building blocks and returns the fitness of the element.

        Parameters
        ----------
            element: list
            list of building blocks
            multiFidelity: bool
            if True, the function will return the fitness and the fidelity of the element

        Returns
        -------
            float
            the fitness of the element
            str
            the identifier of the element

        """
        for x in element:
            if type(x) == int or type(x) == np.float64:
                return float(x), "test"
        return None


class Look_up_table:
    """Class for look up table objective functions
    The look up table objective function is used to evaluate the fitness of the elements by looking up the fitness in a database.

    """

    def __init__(self, df_look_up, fragment_size, target_name="target", aim=0):
        """Initialises the look up table objective function.

        Parameters
        ----------
            df_look_up: pd.DataFrame
            the dataframe containing the look up table
            the dataframe should contain the InChIKeys of the fragments in the form of 'InChIKey_0', 'InChIKey_1', etc.
            and the target column
            and the InChIKeys of the molecule

            fragment_size: int
            the size of the fragments

            target_name: str
            the name of the target column

            aim: int or float
            the aim of the fitness function
            if the aim is an int, the fitness function will be the negative absolute difference between the target and the aim

        """
        self.df_look_up = df_look_up
        self.fragment_size = fragment_size
        self.target_name = target_name
        self.aim = aim
        self.check_database()

    def check_database(self):
        """Checks the database."""
        if self.df_look_up is None:
            msg = "No database found"
            raise ValueError(msg)
        if "InChIKey" not in self.df_look_up.columns:
            msg = "No InChIKey column found"
            raise ValueError(msg)
        if self.target_name not in self.df_look_up.columns:
            msg = "No target column found"
            raise ValueError(msg)
        if any(f"InChIKey_{i}" not in self.df_look_up.columns for i in range(self.fragment_size)):
            msg = "No fragment columns found or not enough fragment columns"
            raise ValueError(
                msg
            )

    def evaluate_element(self, element, multiFidelity=False):
        """Evaluates the fitness of the element
        takes as an input a list of building blocks and returns the fitness of the element.

        Parameters
        ----------
            element: list
            list of building blocks
            multiFidelity: bool
            if True, the function will return the fitness and the fidelity of the element

        Returns
        -------
            float
            the fitness of the element
            str
            the identifier of the element in the form of an InChIKey

        """
        columns = [f"InChIKey_{i}" for i in range(self.fragment_size)]
        if multiFidelity:
            columns.append("fidelity")
        results = element.merge(
            self.df_look_up,
            on=columns,
            how="left",
        )

        results = results.drop_duplicates(
            subset=[f"InChIKey_{i}" for i in range(self.fragment_size)],
        )
        if results[self.target_name].isna().any():
            msg = "missing data"
            raise ValueError(msg)
        if isinstance(self.aim, (int, float)):
            target = -np.abs(results[self.target_name][0] - self.aim)
        else:
            target = results[self.target_name][0]
        return target, results["InChIKey"][0]


class IP_ES1_fosc(Objective_Function):
    """Class for the IP_ES1_fosc objective function
    The IP_ES1_fosc objective function is used to evaluate the fitness of the molecules by calculating the ionisation potential,
    the first excited state energy and the first excited state oscillator strength.
    The fitness function is defined as:
    -np.abs(IP - 5.5) - 0.5 * np.abs(Es1 - 3) + np.log10
    where IP is the ionisation potential, Es1 is the first excited state energy and fosc_1 is the first excited state oscillator strength
    Here the quantum chemical calculation are done using xtb and stda.

    Functions
    ---------
    evaluate_element(element, multiFidelity=False)
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
    run_stda(polymer, STDA_bin_path, output_dir, property, state, database, collection, client)
        Runs the stda calculation of the excited state energy and oscillator strength
        takes as an input the polymer, the path to stda, the output directory, the property, the state, the database and collection name and the client
        returns the excited state energy or oscillator strength



    """

    def __init__(
        self,
        oligomer_size,
        client="mongodb://ch-atarzia.ch.ic.ac.uk/",
        db_mol="stk_mohammed",
        xtb_path="/rds/general/user/ma11115/home/anaconda3/envs/ML/bin/xtb",
        STDA_bin_path="/rds/general/user/ma11115/home/bin/stda_files/xtb4stda/",
        Db_folder="/rds/general/ephemeral/user/ma11115/ephemeral/BO_polymers",
        database_new_calc="stk_mohammed_BO",
        collection_name=None,
        host_IP="cx1",
    ):
        """Initialises the IP_ES1_fosc objective function.

        Parameters
        ----------
            oligomer_size: int
            the size of the oligomer
            client: str
            the path to the mongodb client
            db_mol: str
            the name of the database containing the building blocks
            the database should contain the building blocks position matrix and the InChIKey
            It is normally generated using stk and the stk.MoleculeMongoDb class
            xtb_path: str
            the path to the xtb executable
            STDA_bin_path: str
            the path to the stda executable
            Db_folder: str
            the path to the output directory
            database_new_calc: str
            the name of the database containing the new calculations
            collection_name: str
            the name of the collection
            host_IP: str
            the host IP

        """
        self.client = client
        self.db_mol = db_mol
        self.xtb_path = xtb_path
        self.STDA_bin_path = STDA_bin_path
        self.Db_folder = Db_folder
        os.makedirs(self.Db_folder, exist_ok=True)
        self.database_new_calc = database_new_calc
        self.collection_name = collection_name
        if self.collection_name is None:
            self.collection_name = f"BO_{oligomer_size}"
        self.host_IP = host_IP
        self.oligomer_size = oligomer_size
        self.test_mongo_db_connection()
        self.test_xtb_stda_connection()

    def test_mongo_db_connection(self):
        """Tests the connection to the database."""
        try:
            client = pymongo.MongoClient(self.client)
            stk.MoleculeMongoDb(
                client,
                database=self.db_mol,
            )
        except Exception:
            pass

    def test_xtb_stda_connection(self):
        """Tests the connection to xtb and stda."""
        try:
            os.system(self.xtb_path + " --version")
            os.system(self.STDA_bin_path + " --version")
        except Exception:
            pass

    def evaluate_element(self, element, multiFidelity=False):
        """Evaluates the fitness of the element
        takes as an input a list of building blocks and returns the fitness of the element
        The evaluation here is done by first building the polymer from the building blocks
        then running the xtb optimisation, the xtb calculation of the ionisation potential and the stda calculation of the excited state energy and oscillator strength
        The fitness function is defined as:
        -np.abs(IP - 5.5) - 0.5 * np.abs(Es1 - 3) + np.log10(fosc_1 + 1e-10)
        where IP is the ionisation potential, Es1 is the first excited state energy and fosc_1 is the first excited state oscillator strength.

        Parameters
        ----------
            element: list
            list of building blocks
            multiFidelity: bool
            if True, the function will return the fitness and the fidelity of the element

        Returns
        -------
            float
            the fitness of the element
            str
            the identifier of the element in the form of an InChIKey

        """
        # initialise the database
        client = pymongo.MongoClient(self.client)
        db_mol = stk.MoleculeMongoDb(
            client,
            database=self.db_mol,
        )
        # define the path to xtb and stda
        xtb_path = self.xtb_path
        STDA_bin_path = self.STDA_bin_path
        # define the output directories
        Db_folder = self.Db_folder
        output_dir_ipea = os.path.join(
            Db_folder, "Database", "xtb_calculations"
        )
        xtb_opt_output_dir = os.path.join(
            Db_folder, "Database", "xtb_opt_output_dir"
        )
        output_dir_stda = os.path.join(
            Db_folder, "Database", "stda_output_dir"
        )
        os.makedirs(output_dir_ipea, exist_ok=True)
        os.makedirs(xtb_opt_output_dir, exist_ok=True)
        os.makedirs(output_dir_stda, exist_ok=True)
        # define the database and collection name
        database_new_calc = self.database_new_calc
        collection_name = self.collection_name
        # print(collection_name)
        # build the polymer
        polymer = self.Build_polymer(element, db=db_mol)
        polymer = self.run_xtb_opt(
            polymer,
            xtb_path,
            xtb_opt_output_dir,
            database=database_new_calc,
            collection=collection_name + "_opt",
            client=client,
        )
        Inchikey = stk.InchiKey().get_key(polymer)

        IP = self.run_xtb_ipea(
            polymer,
            xtb_path,
            output_dir_ipea,
            database=database_new_calc,
            target="ionisation potential (eV)",
            collection=collection_name + "_IPEA",
            client=client,
        )
        Es1 = self.run_stda(
            polymer,
            STDA_bin_path,
            output_dir_stda,
            property="Excited state energy (eV)",
            state=0,
            database=database_new_calc,
            collection=collection_name + "_Stda",
            client=client,
        )
        fosc_1 = self.run_stda(
            polymer,
            STDA_bin_path,
            output_dir_stda,
            property="Excited state oscillator strength",
            state=0,
            database=database_new_calc,
            collection=collection_name + "_Stda",
            client=client,
        )
        fitness_function = (
            -np.abs(IP - 5.5)
            - 0.5 * np.abs(Es1 - 3)
            + np.log10(fosc_1 + 1e-10)
        )
        return fitness_function, Inchikey

    def Build_polymer(
        self, element: pd.DataFrame, db: stk.MoleculeMongoDb = None
    ):
        """Builds the polymer from the building blocks
        takes as an input a list of building blocks and a database containing the building blocks
        returns the polymer.

        Parameters
        ----------
            element: pd.DataFrame
            the dataframe containing the building blocks
            db: stk.MoleculeMongoDb
            the database containing the building blocks

        Returns
        -------
            stk.ConstructedMolecule
            the polymer

        """
        precursors = []
        genes = "ABCDEFGH"
        genes = genes[: self.oligomer_size]
        # print(genes)
        repeating_unit = ""
        # joins the Genes to make a repeating unit string
        repeating_unit = repeating_unit.join(genes)
        InchiKey_cols = [col for col in element.columns if "InChIKey_" in col]
        # print(element[InchiKey_cols].values.flatten())
        for fragment in element[InchiKey_cols].values.flatten():
            mol = db.get({"InChIKey": fragment})
            bb = stk.BuildingBlock.init_from_molecule(
                mol, functional_groups=[stk.BromoFactory()]
            )
            precursors.append(bb)
        return stk.ConstructedMolecule(
            stk.polymer.Linear(
                building_blocks=precursors,
                repeating_unit=repeating_unit,
                num_repeating_units=1,
                # optimizer=stk.MCHammer()
            )
        )

    def run_xtb_opt(
        self,
        polymer,
        xtb_path,
        xtb_opt_output_dir,
        database="stk_mohammed_BO",
        collection="test",
        client=None,
    ):
        """Runs the xtb optimisation of the polymer.

        Parameters
        ----------
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

        Returns
        -------
            stk.ConstructedMolecule
            the optimised polymer

        """

        def save_xtb_opt_calculation(
            polymer, xtb_opt_output_dir, collection=None, InchiKey_initial=None
        ) -> None:
            """Saves the xtb optimisation calculation.
            
            Parameters
            ----------
                polymer: stk.ConstructedMolecule
                the polymer
                xtb_opt_output_dir: str
                the output directory
                collection: pymongo.collection
                the collection
                InchiKey_initial: str
                the initial InChIKey
                
            
            Returns
            -------
            None

            """
            def get_property_value(data, property_name):
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
                "cal_folder": os.path.join(
                    xtb_opt_output_dir, stk.InchiKey().get_key(polymer)
                ),
                "Host IP": self.host_IP,
                "InChIKey_initial": InchiKey_initial,
            }
            outfile = open(
                os.path.join(
                    polymer_xtb_opt_calc["cal_folder"], "optimization_1.output"
                ),
                encoding="utf8",
            )
            data = outfile.readlines()
            outfile.close()
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

        collection = client[database][collection]
        if (
            collection.find_one({"InChIKey": get_inchi_key(polymer)})
            is not None
        ):
            # print("already calculated", end="\r")

            db_polymer = stk.ConstructedMoleculeMongoDb(
                client,
                database=database,
            )
            return db_polymer.get({"InChIKey": get_inchi_key(polymer)})
            # print(get_inchi_key(polymer), ' opt geom already calculated')
        if (
            collection.find_one({"InChIKey_initial": get_inchi_key(polymer)})
            is not None
        ):
            # print("already calculated", end="\r")
            db_polymer = stk.ConstructedMoleculeMongoDb(
                client,
                database=database,
            )
            data = collection.find_one(
                {"InChIKey_initial": get_inchi_key(polymer)}
            )
            # print(get_inchi_key(polymer), ' opt geom already calculated with old geom')

            return db_polymer.get({"InChIKey": data["InChIKey"]})
        output_dir = os.path.join(xtb_opt_output_dir, get_inchi_key(polymer))
        InchiKey_initial = get_inchi_key(polymer)
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
        new_output_dir = os.path.join(
            xtb_opt_output_dir, get_inchi_key(polymer)
        )
        os.rename(output_dir, new_output_dir)
        save_xtb_opt_calculation(
            polymer,
            xtb_opt_output_dir,
            collection=collection,
            InchiKey_initial=InchiKey_initial,
        )
        db_polymer = stk.ConstructedMoleculeMongoDb(
            client,
            database=database,
        )
        db_polymer.put(polymer)
        return polymer

    def run_xtb_ipea(
        self,
        polymer,
        xtb_path,
        xtb_opt_output_dir,
        database="stk_mohammed_BO",
        collection="testIPEA",
        target="ionisation potential (eV)",
        client=None,
    ):
        """Runs the xtb calculation of the ionisation potential.

        Parameters
        ----------
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
            target: str
            the target
            client: pymongo.MongoClient
            the client

        Returns
        -------
            float
            the ionisation potential

        """
        collection = client[database][collection]
        XTB_results = collection.find_one({"InChIKey": get_inchi_key(polymer)})
        if XTB_results is not None:
            # print("already calculated", end="\r")
            # print(get_inchi_key(polymer), ' ipea geom already calculated')
            return XTB_results[target]
        xtb = XTBEnergy2(
            xtb_path=xtb_path,
            output_dir=os.path.join(
                xtb_opt_output_dir, get_inchi_key(polymer)
            ),
            unlimited_memory=False,
            calculate_ip_and_ea=True,
            num_cores=25,
        )
        xtb_results = xtb.get_results(polymer)
        XTB_results = {
            "total energy (au)": xtb_results.get_total_energy()[0],
            "homo lumo_gap (eV)": xtb_results.get_homo_lumo_gap()[0],
            "electron affinity (eV)": xtb_results.get_electron_affinity()[0],
            "ionisation potential (eV)": xtb_results.get_ionisation_potential()[
                0
            ],
            "InChIKey": get_inchi_key(polymer),
            "cal_folder": os.path.join(
                xtb_opt_output_dir, get_inchi_key(polymer)
            ),
            "Host IP": self.host_IP,
        }
        collection.update_many(
            filter={"InChIKey": get_inchi_key(polymer)},
            update={"$set": XTB_results},
            upsert=True,
        )
        return XTB_results[target]

    def run_stda(
        self,
        polymer,
        STDA_bin_path,
        output_dir,
        property="Excited state energy (eV)",
        state=1,
        database="stk_mohammed",
        collection="test",
        client=None,
    ):
        """Run XTB-stda.

        Parameters
        ----------
            polymer: stk.ConstructedMolecule
            the polymer
            STDA_bin_path: str
            the path to the STDA xtb executable
            output_dir: str
            the output directory
            Properte: str   
            the property to output 
            state: int
            the excited state for which we output the property
            database: str
            the name of the database
            collection: str
            the name of the collection
            target: str
            the target
            client: pymongo.MongoClient
            the client

        Returns
        -------
            float
            The property of interrest

        """
        collection = client[database][collection]
        STDA_results = collection.find_one(
            {"InChIKey": get_inchi_key(polymer)}
        )
        if STDA_results is not None:
            # print(get_inchi_key(polymer), ' stda geom already calculated')
            # print(STDA_results[property][state])
            return STDA_results[property][state]
        stda = sTDA_XTB(
            STDA_bin_path=STDA_bin_path,
            Num_threads=25,
            output_dir=os.path.join(output_dir, get_inchi_key(polymer)),
        )
        Excited_state_energy, Excited_state_osc = stda.get_results(polymer)
        STDA_results = {
            "Excited state energy (eV)": Excited_state_energy,
            "Excited state oscillator strength": Excited_state_osc,
            "InChIKey": get_inchi_key(polymer),
            "cal_folder": os.path.join(output_dir, get_inchi_key(polymer)),
            "Host IP": self.host_IP,
        }
        collection.update_many(
            filter={"InChIKey": get_inchi_key(polymer)},
            update={"$set": STDA_results},
            upsert=True,
        )
        return STDA_results[property][state]
