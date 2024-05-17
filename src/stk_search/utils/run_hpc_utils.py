
import os
import datetime


def generate_string_run(
        
    case = 'BO_precursor',
    test_name = f"test",
    target = 'target', aim = 0,
    benchmark = True,
    num_iteration = 2,
    num_elem_initialisation = 10,
    which_acquisition = 'EI',
    dataset_representation_path='',
    df_path = "data/output/Full_dataset/df_total_2024-01-05.csv",
    df_precursors_path = "data/output/Prescursor_data/calculation_data_precursor_190923_clean.pkl",
    config_dir = "/rds/general/user/ma11115/home/Geom3D/Geom3D/training/SchNet_frag/",
    search_space_loc = "/rds/general/user/ma11115/home/STK_Search/STK_search/data/input/search_space/test/search_space1.pkl",
    frag_properties='all',
    lim_counter = 10,
    ):
    """ 
    Generate the string to run the search notebook
    
    Args:
        case: str
            case name
        test_name: str
            test name
        target: str
            target name
        aim: int
            aim
        benchmark: bool
            benchmark
        num_iteration: int
            num_iteration
        num_elem_initialisation: int
            num_elem_initialisation
        which_acquisition: str
            which_acquisition
        dataset_representation_path: str
            dataset_representation_path
        df_path: str
            df_path
        df_precursors_path: str
            df_precursors_path
        config_dir: str
            config_dir
        search_space_loc: str
            search_space_loc
    Returns:
        string_to_run_notbook: str
            string to run the notebook
        script_qsub: str
            script to run the notebook on HPC
            
    """
    input = locals()

    string_to_run_notbook = 'src/dev_scripts/run_search_new.py '
    if benchmark:
        test_name = f"benchmark/{test_name}"
    for key, value in input.items():
        if value==True:
            string_to_run_notbook = f"{string_to_run_notbook} --{key} 1"
        elif value==False:
            pass
        else:
            if value != '':
                string_to_run_notbook = f"{string_to_run_notbook} --{key} {value}"
    string_to_run = f"python {string_to_run_notbook} "
    if benchmark:
        num_cpus, mem = 8 , 24
        num_iterations = 50
    else:
        num_cpus, mem = 30 , 50
        num_iterations = 20
    script_qsub = "#!/bin/bash \n"+\
                    "#PBS -l walltime=07:59:01 \n"+\
                    f"#PBS -l select=1:ncpus={num_cpus}:mem={mem}gb:avx=true \n"+\
                    f"#PBS -J 1-{num_iterations} \n"+\
                    " \n"+\
                    "cd /rds/general/user/ma11115/home/STK_Search/STK_search \n"+\
                    "module load anaconda3/personal \n"+ \
                    "source activate Geom3D     \n"+\
                    string_to_run
    print(string_to_run_notbook)
    return string_to_run_notbook, script_qsub

def submit_job(script_qsub, case_name):

    now = datetime.datetime.now()
    # print (now.strftime("%Y-%m-%d %H:%M:%S"))
    sh_file_name = (
        f"HPC_bash_script/Runsearch_{case_name}_{now.strftime('%M_%S_%f')}.sh"
    )
    with open(sh_file_name, "w") as text_file:
        text_file.write(script_qsub)

    os.system(f"qsub -e ./cache -o ./cache {sh_file_name}")
    return