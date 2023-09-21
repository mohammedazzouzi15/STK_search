import os
import sys

import numpy as np

sys.path.append("/rds/general/user/ma11115/home/SDK_EA_MO/Scripts/")

#!/usr/bin/env python3
import psutil
import pymongo
import qml
import stk
from qml.representations import get_slatm_mbtypes


def generate_Slatm(df_1, dirname, name, database_name="stk_mohammed_new"):
    """generate slatm representation following the script in :
    https://github.com/lcmd-epfl/FORMED_ML/blob/a5d1e588dbb4883de19d4a69fae6694b9bde1101/data/generate_slatm.py
    """
    client = pymongo.MongoClient("mongodb://129.31.66.201/")
    os.makedirs(dirname, exist_ok=True)
    db = stk.MoleculeMongoDb(
        client,
        database=database_name,
    )
    namelist = []
    for inchkey in set(df_1["InChIKey"]):
        try:
            if not os.path.exists("cache/{inchkey}.xyz"):
                polymer = db.get({"InChIKey": inchkey})
                polymer.write(f"cache/{inchkey}.xyz")
            # mols_1.append(ase.io.read(f'cache/{inchkey}.xyz'))
            namelist.append(qml.Compound(xyz=f"cache/{inchkey}.xyz"))

        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            print(f" issue with {inchkey}")
    compounds = np.asarray(namelist, dtype=object)  # WARNING: REMOVE SLICING
    print(
        "Generated compounds; RAM memory % used:",
        psutil.virtual_memory()[2],
        flush=True,
    )
    print("Total RAM:", psutil.virtual_memory()[0], flush=True)
    print("Available RAM:", psutil.virtual_memory()[1], flush=True)
    if os.path.exists(dirname + "/mbtypes.npy"):
        mbtypes = np.load(dirname + "/mbtypes.npy", allow_pickle=True)
    else:
        mbtypes = get_slatm_mbtypes([mol.nuclear_charges for mol in compounds])
        mbtypes = np.array(mbtypes)
        np.save(dirname + "/mbtypes.npy", mbtypes)
    for i, mol in enumerate(compounds):
        print(f"Tackling representation of {namelist[i]}", flush=True)
        mol.generate_slatm(mbtypes, local=False, dgrids=[0.1, 0.1])
        print(mol.representation.shape)
        break
    SIZEOFSLATM = len(mol.representation)
    Slatm_array = np.zeros((len(compounds), SIZEOFSLATM), dtype=np.float16)
    N = []
    print(
        "Generated empty representation matrix; RAM memory % used:",
        psutil.virtual_memory()[2],
        flush=True,
    )
    for i, mol in enumerate(compounds):
        print(f"Tackling representation of {namelist[i]}", flush=True)
        mol.generate_slatm(mbtypes, local=False, dgrids=[0.1, 0.1])
        # print(mol.representation.shape)
        Slatm_array[i, :] = np.float16(mol.representation)
        N.append(mol.name)
        print(
            "Filled in one representation vector; RAM memory % used:",
            psutil.virtual_memory()[2],
            flush=True,
        )
        del mol

    N = np.array(N)
    np.save(f"{dirname}/repr_{name}.npy", Slatm_array)
    np.save(f"{dirname}/names_{name}.npy", N)
    return Slatm_array


def generate_Slatm_CM(df_1, dirname, name, database_name="stk_mohammed_BO"):
    """generate slatm representation following the script in :
    https://github.com/lcmd-epfl/FORMED_ML/blob/a5d1e588dbb4883de19d4a69fae6694b9bde1101/data/generate_slatm.py
    """
    client = pymongo.MongoClient("mongodb://129.31.66.201/")
    os.makedirs(dirname, exist_ok=True)
    db = stk.ConstructedMoleculeMongoDb(
        client,
        database=database_name,
    )
    namelist = []
    for inchkey in df_1["InChIKey"]:
        try:
            if not os.path.exists("cache/{inchkey}.xyz"):
                polymer = db.get({"InChIKey": inchkey})
                polymer.write(f"cache/{inchkey}.xyz")
            # mols_1.append(ase.io.read(f'cache/{inchkey}.xyz'))
            namelist.append(qml.Compound(xyz=f"cache/{inchkey}.xyz"))

        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            print(f" issue with {inchkey}")
    compounds = np.asarray(namelist, dtype=object)  # WARNING: REMOVE SLICING
    print(
        "Generated compounds; RAM memory % used:",
        psutil.virtual_memory()[2],
        flush=True,
    )
    print("Total RAM:", psutil.virtual_memory()[0], flush=True)
    print("Available RAM:", psutil.virtual_memory()[1], flush=True)
    if os.path.exists(dirname + "/mbtypes.npy"):
        mbtypes = np.load(dirname + "/mbtypes.npy", allow_pickle=True)
    else:
        mbtypes = get_slatm_mbtypes([mol.nuclear_charges for mol in compounds])
        mbtypes = np.array(mbtypes)
        np.save(dirname + "/mbtypes.npy", mbtypes)
    for i, mol in enumerate(compounds):
        print(f"Tackling representation of {namelist[i]}", flush=True)
        mol.generate_slatm(mbtypes, local=False, dgrids=[0.1, 0.1])
        print(mol.representation.shape)
        break
    SIZEOFSLATM = len(mol.representation)
    Slatm_array = np.zeros((len(compounds), SIZEOFSLATM), dtype=np.float16)
    N = []
    print(
        "Generated empty representation matrix; RAM memory % used:",
        psutil.virtual_memory()[2],
        flush=True,
    )
    for i, mol in enumerate(compounds):
        print(f"Tackling representation of {namelist[i]}", flush=True)
        mol.generate_slatm(mbtypes, local=False, dgrids=[0.1, 0.1])
        # print(mol.representation.shape)
        Slatm_array[i, :] = np.float16(mol.representation)
        N.append(mol.name)
        print(
            "Filled in one representation vector; RAM memory % used:",
            psutil.virtual_memory()[2],
            flush=True,
        )
        del mol

    N = np.array(N)
    np.save(f"{dirname}/repr_{name}.npy", Slatm_array)
    np.save(f"{dirname}/names_{name}.npy", N)
    return Slatm_array
