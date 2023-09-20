
import logging
import os
import shutil
import uuid
import subprocess as sp
from stko.calculators import Calculator
import re



class sTDA_XTB(Calculator):
    def __init__(self, STDA_bin_path, Num_threads=1, output_dir=None,
                 maxeV_ExcitedEnergy=10, property='Energy', state=1):
        self.STDA_bin_path = STDA_bin_path
        self.Num_threads = Num_threads
        self._output_dir = output_dir
        self.maxeV_ExcitedEnergy = maxeV_ExcitedEnergy
        
    def calculate(self, mol):
        if self._output_dir is None:
            output_dir = str(uuid.uuid4().int)
        else:
            output_dir = self._output_dir
        output_dir = os.path.abspath(output_dir)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        init_dir = os.getcwd()
        xyz = os.path.join(output_dir, 'input_structure.xyz')
        mol.write(xyz)
        xyz = os.path.join('input_structure.xyz')
        os.chdir(output_dir)
        sp.call(f"export XTB4STDAHOME={self.STDA_bin_path} \n" +
                f"export OMP_NUM_THREADS={self.Num_threads} \n" +
                f"export MKL_NUM_THREADS={self.Num_threads} \n" +
                "export PATH=$PATH:$XTB4STDAHOME/exe \n" +
                f"xtb4stda {xyz} > gen_wfn.out \n" +
                f"stda_v1.6.2 -xtb -e {self.maxeV_ExcitedEnergy} > out_stda.out",
                shell=True,  stdout=sp.DEVNULL,        stderr=sp.STDOUT)

        os.chdir(init_dir)
        return output_dir

    def get_results(self, mol):

        output_dir = self.calculate(mol)
        init_dir = os.getcwd()
        os.chdir(output_dir)
        outfile = open('out_stda.out', 'r', encoding="utf8")
        data = outfile.readlines()
        outfile.close()
        for i in range(1, len(data)):
            line = data[i]
            if 'state    eV ' in line:
                Excited_state_properties = [re.findall(
                    r"[-+]?(?:\d*\.*\d+)", data[i+x+1]) for x in range(10)]
                Excited_state_energy = [float(x[1]) for x in
                                        Excited_state_properties]  # float(words[3]) #
                Excited_state_osc = [float(x[3]) for x in
                                     Excited_state_properties]


        os.chdir(init_dir)
        if Excited_state_energy == []:
            print('Error: No Excited state properties found')
        return Excited_state_energy ,Excited_state_osc
