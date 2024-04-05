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
    def __init__(self):
        pass

    def evaluate_element(self, element):
        for x in element:
            if type(x) == int or type(x) == np.float64:
                return float(x), "test"


class Look_up_table:
    def __init__(self, df_look_up, fragment_size, target_name="target", aim=0):
        self.df_look_up = df_look_up
        self.fragment_size = fragment_size
        self.target_name = target_name
        self.aim = aim

    def evaluate_element(self, element):
        # if type(element) == pd.Series:
        # element = element.to_frame()
        results = element.merge(
            self.df_look_up,
            on=[f"InChIKey_{i}" for i in range(self.fragment_size)],
            how="left",
        )

        results.drop_duplicates(
            subset=[f"InChIKey_{i}" for i in range(self.fragment_size)],
            inplace=True,
        )
        if results[self.target_name].isna().any():
            print("missing data")
            raise ValueError("missing data")
        if isinstance(self.aim, (int, float)):
            target = -np.abs(results[self.target_name][0] - self.aim)
        else:
            target = results[self.target_name][0] 
        return target, results["InChIKey"][0]


class IP_ES1_fosc(Objective_Function):
    def __init__(self, oligomer_size):
        self.client = "mongodb://ch-atarzia.ch.ic.ac.uk/"
        self.db_mol = "stk_mohammed_new"
        self.xtb_path = (
            "/rds/general/user/ma11115/home/anaconda3/envs/ML/bin/xtb"
        )

        self.STDA_bin_path = (
            "/rds/general/user/ma11115/home/bin/stda_files/xtb4stda/"
        )
        self.Db_folder = (
            "/rds/general/ephemeral/user/ma11115/ephemeral/BO_polymers"
        )
        os.makedirs(self.Db_folder, exist_ok=True)
        self.database_new_calc = "stk_mohammed_BO"
        if oligomer_size == 6:
            self.collection_name = "BO_exp1"
        else:
            self.collection_name = f"BO_{oligomer_size}"
        # print(self.collection_name)
        self.host_IP = "cx1"
        self.oligomer_size = oligomer_size

    def evaluate_element(self, element):
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
            database="stk_mohammed_BO",
            collection=collection_name + "_Stda",
            client=client,
        )
        fosc_1 = self.run_stda(
            polymer,
            STDA_bin_path,
            output_dir_stda,
            property="Excited state oscillator strength",
            state=0,
            database="stk_mohammed_BO",
            collection=collection_name + "_Stda",
            client=client,
        )
        fitness_function = (
            -np.abs(IP - 5.5) - 0.5 * np.abs(Es1 - 3) + np.log10(fosc_1+1e-10)
        )
        return fitness_function, Inchikey

    def Build_polymer(
        self, element: pd.DataFrame, db: stk.MoleculeMongoDb = None
    ):
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
        polymer = stk.ConstructedMolecule(
            stk.polymer.Linear(
                building_blocks=precursors,
                repeating_unit=repeating_unit,
                num_repeating_units=1,
                # optimizer=stk.MCHammer()
            )
        )
        return polymer

    def run_xtb_opt(
        self,
        polymer,
        xtb_path,
        xtb_opt_output_dir,
        database="stk_mohammed_BO",
        collection="test",
        client=None,
    ):
        def save_xtb_opt_calculation(
            polymer, xtb_opt_output_dir, collection=None, InchiKey_initial=None
        ):
            def get_property_value(data, property_name):
                for line in data:
                    if property_name in line:
                        if property_name == "cpu-time":
                            property_value = (
                                re.findall(r"[-+]?(?:\d*\.*\d+)", line)[-3]
                                + " h "
                                + re.findall(r"[-+]?(?:\d*\.*\d+)", line)[-2]
                                + " min "
                                + re.findall(r"[-+]?(?:\d*\.*\d+)", line)[-1]
                                + " s "
                            )
                            return property_value
                        property_value = float(
                            re.findall(r"[-+]?(?:\d*\.*\d+)", line)[-1]
                        )  # float(words[3]) #
                        return property_value

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
                "r",
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
            polymer = db_polymer.get({"InChIKey": get_inchi_key(polymer)})
            # print(get_inchi_key(polymer), ' opt geom already calculated')
            return polymer
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

            polymer = db_polymer.get({"InChIKey": data["InChIKey"]})
            return polymer
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
