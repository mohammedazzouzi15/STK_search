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

class Calculate_Precursor():
    def __init__(self):
        self.client = "mongodb://ch-atarzia.ch.ic.ac.uk/"
        self.db_mol = "stk_mohammed_new"
        self.xtb_path = (
            "/rds/general/user/ma11115/home/anaconda3/envs/ML/bin/xtb"
        )

        self.Db_folder = (
            "/rds/general/ephemeral/user/ma11115/ephemeral/BO_precursor"
        )
        os.makedirs(self.Db_folder, exist_ok=True)
        # print(self.collection_name)
        self.host_IP = "cx1"
        self.collection_name = "Precursors"

    def load_precursors(self, smile):
        """ Function to generate stk building block from smiles"""
        precursor = stk.BuildingBlock(
                smile, functional_groups=[
                    stk.BromoFactory()])

        return precursor

    def evaluate_element(self, smile):
        """function to evaluate the element
        depending on the paths provided (xtb or stda )
        the function will add those calculations to the model"""
        # initialise the database
        client = pymongo.MongoClient(self.client)
        db_mol = stk.MoleculeMongoDb(
            client,
            database=self.db_mol,
        )
        # define the path to xtb and stda
        xtb_path = self.xtb_path
        
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
        collection_name = self.collection_name
        # print(collection_name)
        precursor = self.load_precursors(smile)
        if self.xtb_path is not None:
            precursor = self.run_xtb_opt(
                precursor,
                xtb_path,
                xtb_opt_output_dir,
                database=self.db_mol,
                collection=collection_name + "_opt",
                client=client,
            )
            Inchikey = stk.InchiKey().get_key(precursor)

            IP = self.run_xtb_ipea(
                precursor,
                xtb_path,
                output_dir_ipea,
                database=self.db_mol,
                target="ionisation potential (eV)",
                collection=collection_name + "_IPEA",
                client=client,
            )
            if self.STDA_bin_path is not None:
                STDA_bin_path = self.STDA_bin_path
                Es1 = self.run_stda(
                    precursor,
                    STDA_bin_path,
                    output_dir_stda,
                    property="Excited state energy (eV)",
                    state=0,
                    database=self.db_mol,
                    collection=collection_name + "_Stda",
                    client=client,
                )
                return Es1, Inchikey
            return IP, Inchikey
        else: 
            precursor = self.run_ETKDG_opt(
                precursor,
                xtb_opt_output_dir,
                database=self.db_mol,
                client=client,
            )
            Inchikey = stk.InchiKey().get_key(precursor)
            return None, Inchikey
        

    def run_ETKDG_opt(
        self,
        polymer,
        xtb_opt_output_dir,
        database="stk_mohammed_BO",
        client=None,
    ):
        output_dir = os.path.join(xtb_opt_output_dir, get_inchi_key(polymer))
        InchiKey_initial = get_inchi_key(polymer)
        ETKDG = stko.OptimizerSequence(
            stko.ETKDG(),
        )
        polymer = ETKDG.optimize(polymer)
        db_polymer = stk.MoleculeMongoDb(
            client,
            database=database,
        )
        db_polymer.put(polymer)
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

            db_polymer = stk.MoleculeMongoDb(
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
            db_polymer = stk.MoleculeMongoDb(
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
        if ~os.path.exists(new_output_dir):
            os.rename(output_dir, new_output_dir)
        save_xtb_opt_calculation(
            polymer,
            xtb_opt_output_dir,
            collection=collection,
            InchiKey_initial=InchiKey_initial,
        )
        db_polymer = stk.MoleculeMongoDb(
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
