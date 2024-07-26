==========
STK_search
==========

to install cd to the directory 
then 
once you are using the right environment 
to install the package, 

conda create -n stk_search python=3.10

conda activate stk_search

pip install -e . # on the folder where pyproject.toml is located

then install the following packages

pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html


for cuda users: 

pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
