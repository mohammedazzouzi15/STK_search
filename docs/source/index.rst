Welcome to stk_search's documentation!
===================================

skt_search is a Python library for searching the chemical space of molecules formed by stk. It is built on top of stk and stk_optim.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Calculators <_autosummary/stk_search.Calculators>
   utils <_autosummary/stk_search.utils>
   Search_algorithm <_autosummary/stk_search.Search_algorithm>
   Representation <_autosummary/stk_search.Representation>
   SearchAlgorithm <_autosummary/stk_search.Search_algorithm>
   SearchSpace <_autosummary/stk_search.SearchSpace>
   Modules <modules>


.. tip::

  ⭐ Star us on GitHub! ⭐

GitHub: https://GitHub.com/mohammedazzouzi15/stk_search

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
for the usage of the package, please refer to the example notebooks in the notebooks folder.

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

