import os
import shutil
import subprocess as sp
import uuid

import stko


class XTBEnergy2(stko.XTBEnergy):
    def _run_xtb(self, xyz, out_file, init_dir, output_dir):
        """
        Runs GFN-xTB.

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
        if self._unlimited_memory:
            memory = "ulimit -s unlimited ;"
        else:
            memory = ""

        if self._solvent is not None:
            solvent = f"--{self._solvent_model} {self._solvent} "
        else:
            solvent = ""

        if self._calculate_free_energy:
            hess = "--hess"
        else:
            hess = ""

        if self._calculate_ip_and_ea:
            vipea = "--vipea"
        else:
            vipea = ""

        cmd = (
            f"{memory} {self._xtb_path} "
            f"{xyz} --gfn {self._gfn_version} "
            f"{hess} {vipea} --parallel {self._num_cores} "
            f"--etemp {self._electronic_temperature} "
            f"{solvent} --chrg {self._charge} "
            f"--uhf {self._num_unpaired_electrons} -I det_control.in"
        )
        # print(cmd)
        try:
            os.chdir(output_dir)
            self._write_detailed_control()
            with open(out_file, "w") as f:
                # Note that sp.call will hold the program until
                # completion of the calculation.
                sp.call(
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
        if self._output_dir is None:
            output_dir = str(uuid.uuid4().int)
        else:
            output_dir = self._output_dir
        # output_dir = os.path.abspath(output_dir)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        init_dir = os.getcwd()
        xyz = os.path.join(output_dir, "input_structure.xyz")
        out_file = os.path.join("energy.output")
        mol.write(xyz)
        xyz = os.path.join("input_structure.xyz")

        yield self._run_xtb(
            xyz=xyz,
            out_file=out_file,
            init_dir=init_dir,
            output_dir=output_dir,
        )

    def get_results(self, mol):
        """
        Calculate the xTB properties of `mol`.

        Parameters
        ----------
        mol : :class:`.Molecule`
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
        output_dir = os.path.abspath(output_dir)

        out_file = os.path.join(output_dir, "energy.output")

        return stko.XTBResults(
            generator=self.calculate(mol),
            output_file=out_file,
        )