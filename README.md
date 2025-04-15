# Ant Colony Optimization with CUDA

This repository contains the source-code for the Ant Colony Optimization applied to the Traveling Salesman Problem written in CUDA.
Code was written as a project for the HPC course @ MIMUW (2024/2025). Code reproduces the idea introduced
[in this paper](https://doi.org/10.1016/j.jpdc.2012.01.002), where all methods and parameters are explained.

## Input
Program accepts the files in TSP format, definition of which can be found [here](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/).
It should be run with following format 
```
./acotsp <input_file> <output_file> <TYPE> <NUM_ITER> <ALPHA> <BETA> <EVAPORATE> <SEED>

```
where
1. input_file -> TSP input,
2. output_file -> output file for the answer with format: first line - total distance, second line - visited cities.
3. TYPE -> either WORKER or QUEEN (different algorithms),
4. NUM_ITER -> number of iterations to converge,
5. ALPHA,BETA -> hyperparameters for pheromones,
6. EVAPORATE -> rate at which the pheromones evaporate,
7. SEED -> seed used for the simulations.

# Tests
You can run "run_all.py" program that will automatically test the program using some of the test problems with solutions, 
all of which were taken from this [github repository](https://github.com/mastqe/tsplib/blob/master/solutions).