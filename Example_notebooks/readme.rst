notebooks showing how to use the library. 

0. You can add the database used in the original work using the 00_load_database notebook.

1. the first notebook shows how to generate  list of building blocks from a list of smiles. 
    Here we use the function `generate_building_blocks` to generate a list of building blocks from a list of smiles.
    in this case we consider that the connection point between the building blocks is determined by the position of bromine atoms. 
    to build the oligomers we use stk with the stk.bromofactory to assemble the building block. 
    
    If you want to use a different way to attach the building block, you would just need to change the evaluation function where you go from building blocks to 
    molecules. 

2. the second notebook shows how to initialise the search space using the building block dataframe and a precalculated dataframe
    especially how to define conditions on the way to assemble the building blocks using a syntax or conditions on the properties of the building blocks
    as different position.

3. the third notebook shows how to run the search experiment, 
    in this case we use a bayesian optimisation with representation based on the fragment properties

4. the fifth notebook shows how we can train a GNN model to generate either representation or to be used in a bayesian optimisation
