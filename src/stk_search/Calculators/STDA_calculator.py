"""A module for calculating the excited state properties using sTDA method
from xtb output.


This module provides a class to calculate the excited state properties using
sTDA method from xtb output.

Example:
-------
    To use this module, import it and create an instance of the class. Then
    use the calculate method to calculate the excited state properties.
    
    .. code-block:: python
    
        from stk_search.Calculators.STDA_calculator import sTDA_XTB
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        rdDepictor.Compute2DCoords(mol)
        
        STDA_bin_path = "/path/to/STDA_binary"
        Num_threads = 1
        output_dir = None
        stda_calculator = sTDA_XTB(STDA_bin_path, Num_threads, output_dir)
        Excited_state_energy, Excited_state_osc = stda_calculator.get_results(mol)
        print(Excited_state_energy)
        print(Excited_state_osc)

        
"""  # noqa: D205

import os
import re
import shutil
import subprocess as sp
import uuid



class sTDA_XTB:
    """A class to calculate the excited state properties using sTDA method from xtb output.

    Attributes
    ----------
    STDA_bin_path : str
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
        
        Parameters
        ----------
        STDA_bin_path : str
            The path to the STDA binary file.
        Num_threads : int
            The number of threads to use.
        output_dir : str
            The path to the output directory.
        maxeV_ExcitedEnergy : float
            The maximum energy of the excited state.

        """
        self.stda_bin_path = stda_bin_path
        self.num_threads = num_threads
        self._output_dir = output_dir
        self.maxev_excitedenergy = maxev_excitedenergy

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
        output_dir = os.path.abspath(output_dir)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        init_dir = os.getcwd()
        xyz = os.path.join(output_dir, "input_structure.xyz")
        mol.write(xyz)
        xyz = os.path.join("input_structure.xyz")
        os.chdir(output_dir)
        sp.call(
            f"export XTB4STDAHOME={self.STDA_bin_path} \n"
            + f"export OMP_NUM_THREADS={self.Num_threads} \n"
            + f"export MKL_NUM_THREADS={self.Num_threads} \n"
            + "export PATH=$PATH:$XTB4STDAHOME/exe \n"
            + f"xtb4stda {xyz} > gen_wfn.out \n"
            + f"stda_v1.6.2 -xtb -e {self.maxeV_ExcitedEnergy} > out_stda.out",
            shell=True,
            stdout=sp.DEVNULL,
            stderr=sp.STDOUT,
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
        init_dir = os.getcwd()
        os.chdir(output_dir)
        outfile = open("out_stda.out", encoding="utf8")
        data = outfile.readlines()
        outfile.close()
        for i in range(1, len(data)):
            line = data[i]
            if "state    eV " in line:
                Excited_state_properties = [
                    re.findall(r"[-+]?(?:\d*\.*\d+)", data[i + x + 1])
                    for x in range(10)
                ]
                Excited_state_energy = [
                    float(x[1]) for x in Excited_state_properties
                ]  # float(words[3]) #
                Excited_state_osc = [
                    float(x[3]) for x in Excited_state_properties
                ]

        os.chdir(init_dir)
        if Excited_state_energy == []:
            pass
        return Excited_state_energy, Excited_state_osc
