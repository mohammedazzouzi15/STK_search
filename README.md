:Author: Mohammed Azzouzi
:Docs: https://stk-search.readthedocs.io

<div align="center">
  <h1>stk-search</h1>
  <h2>Search over the space of molecules built from fragments</h2>
</div>

<p align="center">
  <img src="./overview.svg" alt="Overview Image" />
</p>

## Overview

`stk_search` is a Python package for searching the chemical space of molecules formed by `stk`. It is built on top of `stk` and `stko`. For more details on the use of the package, please refer to the corresponding publication as well as the documentation associated with it.

We use [stk](https://github.com/lukasturcani/stk) and [stko](https://github.com/JelfsMaterialsGroup/stko) for building and calculating the properties of the molecules.

We use [BoTorch](https://botorch.org/) for the implementation of the Bayesian optimization.

For the implementation of the geometric modeling on 3D structure, we use the implementation of models in [GEOM3D](https://github.com/chao1224/Geom3D).

## Installation

To install the package, follow these steps:

1. **Open a terminal and change to the directory where the `pyproject.toml` file is located.**
    ```bash
    cd path/to/directory
    ```

2. **Create a new conda environment**
    ```bash
    conda create -n stk_search python=3.8
    ```

3. **Activate the environment**
    ```bash
    conda activate stk_search
    ```

4. **Run the following command to install the package:**
    In some cases, you may need to install `gcc` before installing the package.
    ```bash
    pip install -e .
    ```

5. **Install additional packages to use the GNN model:**

    **For GPU:**
    ```bash
    pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
    # Make sure the torch version is the right one
    ```

    **For CPU:**
    ```bash
    pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
    ```
## Directory Structure

| Directory | Description |
|-----------|-------------|
| [Example_notebooks]| Contains Jupyter notebooks demonstrating the use of the package. |
| `src/stk_search/algorithms` | Contains the code of the package. |
| [utils]| Utility functions used throughout the package. |
| [docs] | Documentation for the package. |
| [tests] | Contains unit tests for the package. |

## STK_search Directory Structure

| Path | Description |
|------|-------------|
| `src/stk_search/` | Root directory for the STK search module |
| `src/stk_search/SearchSpace.py` | Module defining the SearchSpace class |
| `src/stk_search/SearchExp.py` | Module defining the SearchExp class for running search experiments |
| `src/stk_search/SearchedSpace.py` | Module defining the SearchedSpace class for representing the searched space |
| `src/stk_search/Search_algorithm/` | Contains modules for different search algorithms |
| `src/stk_search/Search_algorithm/Search_algorithm.py` | Base class for search algorithms |
| `src/stk_search/Search_algorithm/MultifidelityBayesianOptimisation.py` | Module for multifidelity Bayesian optimization |
| `src/stk_search/Search_algorithm/Ea_surrogate.py` | Module for evolutionary algorithm with surrogate model |
| `src/stk_search/Search_algorithm/BayesianOptimisation.py` | Module for Bayesian optimization |
| `src/stk_search/Search_algorithm/BayesianOptimisation_ErrPred.py` | Module for Bayesian optimization with error prediction |
| `src/stk_search/Precursors/` | Contains modules for precursor generation and database utilities |
| `src/stk_search/Precursors/precursor_database_utils.py` | Module for precursor database utilities |
| `src/stk_search/ObjectiveFunctions/` | Contains modules for defining objective functions |
| `src/stk_search/ObjectiveFunctions/__init__.py` | Initialization file for the ObjectiveFunctions submodule |
| `src/stk_search/ObjectiveFunctions/ObjectiveFunction.py` | Base class for objective functions |
| `src/stk_search/ObjectiveFunctions/IpEs1Fosc.py` | Module for specific objective function implementation |
| `src/stk_search/geom3d/` | Contains modules for 3D geometry processing and model training |
| `src/stk_search/geom3d/script_plot_inference_BA.py` | Script for plotting inference results |
| `src/stk_search/geom3d/models/` | Contains model definitions for 3D geometry processing |
| `src/stk_search/geom3d/models/Equiformer/` | Contains modules for Equiformer model |
| `src/stk_search/geom3d/train_models.py` | Module for training 3D geometry models |
| `src/stk_search/geom3d/utils/` | Contains utility modules for 3D geometry processing |
| `src/stk_search/geom3d/utils/database_utils.py` | Module for database utilities |
| `src/dev_scripts/` | Contains development scripts for testing and running experiments |
| `src/dev_scripts/run_search.py` | Script for running search experiments |
| `src/dev_scripts/run_search_test.py` | Script for running search experiments with test configurations |
| `src/dev_scripts/run_search_new.py` | Script for running new search experiments |


## Usage

Refer to the example notebooks where we show a step-by-step use of the package to search a space of oligomers formed of 6 building blocks.

1. **Notebook 0: Generate Building Blocks**
    - Shows how to go from a list of SMILES to generate a list of building blocks.
    - Introduces a way to run calculations using `xtb` and `xtb_stda` to get the properties of the building blocks and save them in a database.
    - Demonstrates how to generate a dataframe with the necessary data to form a representation of the constructed molecules for Bayesian optimization.

2. **Notebook 1: Define Search Space**
    - Shows how to define the search space and generate a search space pickle that can be loaded later to run the search algorithm.

3. **Notebook 2: Run Search Algorithm**
    - Shows how to run the search algorithm on the search space using different search algorithms: BO, EA, SUEA.

4. **Notebook 3: Representation Learning**
    - Shows how to run a representation learning using a 3D geometry-based GNN.

## Adding New Search Algorithms

To incorporate new search algorithms into the package, follow these steps:

1. **Create a new Python file for your algorithm in the `src/stk_search/algorithms` directory.**
    ```bash
    touch src/stk_search/algorithms/my_new_algorithm.py
    ```

2. **Implement your search algorithm in the new file.**
    ```python
    # src/stk_search/algorithms/my_new_algorithm.py

    class MyNewAlgorithm:
        def __init__(self, ...):
            # Initialize your algorithm

        def run(self, ...):
            # Implement the logic of your algorithm
    ```

3. **Import and use your new algorithm in the main script or notebook.**
    ```python
    from stk_search.algorithms.my_new_algorithm import MyNewAlgorithm

    # Initialize and run your algorithm
    algorithm = MyNewAlgorithm(...)
    algorithm.run(...)
    ```

4. **Add unit tests for your new algorithm in the [tests](http://_vscodecontentref_/7) directory.**
    ```python
    # tests/test_my_new_algorithm.py

    import unittest
    from stk_search.algorithms.my_new_algorithm import MyNewAlgorithm

    class TestMyNewAlgorithm(unittest.TestCase):
        def test_algorithm(self):
            # Test your algorithm
            algorithm = MyNewAlgorithm(...)
            result = algorithm.run(...)
            self.assertEqual(result, expected_result)

    if __name__ == '__main__':
        unittest.main()
    ```

## Note on Representation Learning

We used the implementation of different GNNs following the code in [Geom3D](https://github.com/chao1224/Geom3D/tree/main). If you use any of the capabilities related to representation learning, please cite their paper:
