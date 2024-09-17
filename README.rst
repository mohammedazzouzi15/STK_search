==========
STK_search
==========

skt_search is a Python library for searching the chemical space of molecules formed by stk. It is built on top of stk and stko. For more details on the use of the package please refer to the corresponding publication as well as the documentation associated with it. 

Installation
============

To install the package, follow these steps:


1. **Open a terminal and change to the directory where the `pyproject.toml` file is located.**
        cd path/to/directory
2. create a new conda environment
        conda create -n stk_search python=3.8 
3. activate the environment
        conda activate stk_search
4. Run the following command to install the package:
In some cases i had to install gcc before installing the package
        pip install -e .
5. install additional package to use the GNN model:
    for GPU:
       pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
       Make sure the torch version is the right one
    for CPU:
        pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html


Usage
============
In order to use the package, you can refer to the example notebooks where we show a step by step use of the package to search a space of oligomers formed of 6 building blocks. 

The first notebook: 0_generate_building_blocks shows how to go from a list of Smile to generate a list of building blocks. 
        In the notebook we also introduce a way to run calculations using xtb and xtb_stda to get the properties of the building blocks and save them in a databse.
        in the last part of the notebook we also show a way to generate a dataframe with the necessary data to form a representation of the constructed molecules if we want to use bayesian optimisation. 

The second notebook shows how to define the search space and generate a search space pickle that we can load later to run the search algorithm 

The third notebook shows how to run the search algorithm on the search space using different search algorithm: BO, EA, SUEA. 

The fourth notebook shows how to run a representation learning using a 3D geometry based GNN. 

Contributing
============
[Provide guidelines for contributing to the project]

License
============
[Specify the license under which the package is distributed]

Contact
============

```
