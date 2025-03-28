"""A calculator that uses the xTB program to calculate energies.

This module defines a calculator that uses the xTB program to calculate
energies. The calculator is a subclass of :class:`.XTBEnergy
<stk.stko.XTBEnergy>`. The calculator is used to calculate the energy
of a molecule.
"""

import os
import shutil
import subprocess as sp
import uuid
from pathlib import Path

import stko


class XTBEnergy2(stko.XTBEnergy):
    """A calculator that uses the xTB program to calculate energies.

    Attributes
    ----------
    xtb_path : :class:`str`
        The path to the xTB executable.

    gfn_version : :class:`int`
        The version of the GFN force field to use.

    num_cores : :class:`int`
        The number of cores to use.

    electronic_temperature : :class:`float`
        The electronic temperature to use.

    charge : :class:`int`
        The charge of the molecule.

    num_unpaired_electrons : :class:`int`
        The number of unpaired electrons in the molecule.

    solvent : :class:`str`
        The solvent to use in the calculation.

    solvent_model : :class:`str`
        The solvent model to use in the calculation.

    calculate_free_energy : :class:`bool`
        Whether to calculate the free energy.

    calculate_ip_and_ea : :class:`bool`
        Whether to calculate the ionization potential and electron
        affinity.

    unlimited_memory : :class:`bool`
        Whether to use unlimited memory.

    output_dir : :class:`str`
        The directory in which to write the output files.

    Methods
    -------
    calculate(mol)
        Calculate the energy of a molecule.
    get_results(mol)
        Get the results of the calculation.

    """

    def _run_xtb(self, xyz, out_file, init_dir, output_dir) -> None:
        """Run the xTB calculation.

        Parameters
        ----------
        xyz : :class:`str`
            The name of the input structure ``.xyz`` file.

        out_file : :class:`str`
            The name of output file with xTB results.

        init_dir : :class:`str`
            The name of the current working directory.

        output_dir : :class:`str`
            The name of the directory into which files generated during
            the calculation are written.

        Returns
        -------
        None : :class:`NoneType`

        """
        # Modify the memory limit.
        memory = "ulimit -s unlimited ;" if self._unlimited_memory else ""

        if self._solvent is not None:
            solvent = f"--{self._solvent_model} {self._solvent} "
        else:
            solvent = ""

        hess = "--hess" if self._calculate_free_energy else ""

        vipea = "--vipea" if self._calculate_ip_and_ea else ""

        cmd = (
            f"{memory} {self._xtb_path} "
            f"{xyz} --gfn {self._gfn_version} "
            f"{hess} {vipea} --parallel {self._num_cores} "
            f"--etemp {self._electronic_temperature} "
            f"{solvent} --chrg {self._charge} "
            f"--uhf {self._num_unpaired_electrons} -I det_control.in"
        )
        try:
            os.chdir(output_dir)
            self._write_detailed_control()
            with Path(out_file).open(mode="w") as f:
                # Note that sp.call will hold the program until
                # completion of the calculation.
                sp.call(  # noqa: S602
                    cmd,
                    stdin=sp.PIPE,
                    stdout=f,
                    stderr=sp.PIPE,
                    # Shell is required to run complex arguments.
                    shell=True,
                )
        finally:
            os.chdir(init_dir)

    def calculate(self, mol):
        """Calculate the xTB energy of a molecule.

        Parameters
        ----------
        mol : :class:`.Molecule`
            The molecule whose energy is to be calculated.

        Yields
        ------
        None

        """
        if self._output_dir is None:
            output_dir = str(uuid.uuid4().int)
        else:
            output_dir = self._output_dir

        if Path(output_dir).exists():
            shutil.rmtree(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        init_dir = Path.cwd()
        xyz = Path(output_dir, "input_structure.xyz")
        out_file =Path("energy.output")
        mol.write(xyz)
        xyz = Path("input_structure.xyz")

        yield self._run_xtb(
            xyz=xyz,
            out_file=out_file,
            init_dir=init_dir,
            output_dir=output_dir,
        )

    def get_results(self, mol):
        """Calculate the xTB properties of a molecule.

        Parameters
        ----------
        mol : :class:`.Molecule`
            The molecule whose properties are to be calculated.
            The :class:`.Molecule` whose energy is to be calculated.

        Returns
        -------
        :class:`.XTBResults`
            The properties, with units, from xTB calculations.

        """
        if self._output_dir is None:
            output_dir = str(uuid.uuid4().int)
        else:
            output_dir = self._output_dir
        output_dir = Path(output_dir).absolute()

        out_file = Path(output_dir, "energy.output")

        return stko.XTBResults(
            generator=self.calculate(mol),
            output_file=out_file,
        )
