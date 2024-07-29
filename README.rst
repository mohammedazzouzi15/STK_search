==========
STK_search
==========

STK_search is a Python package designed for [brief description of what the package does].

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
        pip install -e .
5. install additional package to use the GNN model:
    for GPU:
       pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
       Make sure the torch version is the right one
    for CPU:
        pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html


Usage
============
[Provide examples and explanations on how to use the package]

Contributing
============
[Provide guidelines for contributing to the project]

License
============
[Specify the license under which the package is distributed]

Contact
============

[Provide contact information or links to relevant resources]

```
