"""A module for calculating the excited state properties using sTDA method
from xtb output.


This module provides a class to calculate the excited state properties using
sTDA method from xtb output.

Example:
-------
    To use this module, import it and create an instance of the class. Then
    use the calculate method to calculate the excited state properties.

    .. code-block:: python

        from stk_search.Calculators.stda_calculator import sTDAXTB
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        rdDepictor.Compute2DCoords(mol)

        stda_bin_path = "/path/to/STDA_binary"
        Num_threads = 1
        output_dir = None
        stda_calculator = sTDAXTB(stda_bin_path, Num_threads, output_dir)
        Excited_state_energy, Excited_state_osc = stda_calculator.get_results(mol)
        print(Excited_state_energy)
        print(Excited_state_osc)


"""  # noqa: D205

import logging
import os
import re
import shutil
import subprocess as sp
import uuid
from pathlib import Path


class sTDAXTB:
    """A class to calculate the excited state properties using sTDA method from xtb output.

    Attributes
    ----------
    stda_bin_path : str
        The path to the STDA binary file.
    Num_threads : int
        The number of threads to use.
    output_dir : str
        The path to the output directory.
    maxeV_ExcitedEnergy : float
        The maximum energy of the excited state.

    Methods
    -------
    calculate(mol)
        Calculate the excited state properties.
    get_results(mol)
        Get the results of the calculation.

    """

    def __init__(
        self,
        stda_bin_path,
        num_threads=1,
        output_dir=None,
        maxev_excitedenergy=10,
    ):
        """Initialize the class.

        Args:
        ----
        stda_bin_path : str
            The path to the STDA binary file.
        num_threads : int
            The number of threads to use.
        output_dir : str
            The path to the output directory.
        maxev_excitedenergy : float
            The maximum energy of the excited state.

        """
        self.stda_bin_path = stda_bin_path
        self.num_threads = num_threads
        self._output_dir = output_dir
        self.maxev_excitedenergy = maxev_excitedenergy
        self.XTB4STDAHOME = "/media/mohammed/Work/xtb4stda_home"

    def calculate(self, mol):
        """Calculate the excited state properties.

        Parameters
        ----------
        mol : RDKit Mol
            The molecule to calculate the excited state properties.

        Returns
        -------
        str
            The path to the output directory.

        """
        if self._output_dir is None:
            output_dir = str(uuid.uuid4().int)
        else:
            output_dir = self._output_dir
        if Path(output_dir).exists():
            shutil.rmtree(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        init_dir = Path.cwd()
        xyz = Path(output_dir) / "input_structure.xyz"
        mol.write(xyz)
        xyz = "input_structure.xyz"
        os.chdir(output_dir)
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.num_threads)
        env["MKL_NUM_THREADS"] = str(self.num_threads)
        env["XTB4STDAHOME"] = self.XTB4STDAHOME 
        command = [self.stda_bin_path + "xtb4stda", xyz]
        with Path("gen_wfn.out").open("w", encoding="utf-8") as f:
            sp.run(  # noqa: S603
                command,
                env=env,
                check=True,
                stdout=f,
            )
        command = [
            self.stda_bin_path + "stda_v1.6.3",
            "-xtb",
            "-e",
            str(self.maxev_excitedenergy),
        ]
        with Path("out_stda.out").open("w", encoding="utf-8") as f:
            sp.run(  # noqa: S603
                command,
                env=env,
                check=True,
                stdout=f,
            )

        os.chdir(init_dir)
        return output_dir

    def get_results(self, mol):
        """Get the results of the calculation.

        Parameters
        ----------
        mol : RDKit Mol
            The molecule to calculate the excited state properties.

        Returns
        -------
        list of float
            The excited state energies.
        list of float
            The excited state oscillator strengths.

        """
        output_dir = self.calculate(mol)
        init_dir = Path.cwd()
        os.chdir(output_dir)
        try:
            with Path("out_stda.out").open(
                encoding="utf8", mode="r"
            ) as outfile:
                data = outfile.readlines()
            for i in range(1, len(data)):
                line = data[i]
                if "state    eV " in line:
                    excited_state_properties = [
                        re.findall(r"[-+]?(?:\d*\.*\d+)", data[i + x + 1])
                        for x in range(10)
                    ]
                    excited_state_energy = [
                        float(x[1]) for x in excited_state_properties
                    ]  # float(words[3]) #
                    excited_state_osc = [
                        float(x[3]) for x in excited_state_properties
                    ]
        except FileNotFoundError:
            excited_state_energy = []
            excited_state_osc = []
            logging.exception("No excited state properties found")
        os.chdir(init_dir)
        if excited_state_energy == []:
            pass
        return excited_state_energy, excited_state_osc
