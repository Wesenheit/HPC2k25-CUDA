import argparse
from glob import glob
import os
import time 
from itertools import product
import subprocess

def run_all(args):
    print(args)
    names = glob("examples/*.tsp")
    solutions_dict = {}
    with open("examples/solutions") as f:
        for line in f.readlines():
            name,solution = line.split(":")
            solutions_dict[name.strip()] = int(solution)
    parameters = product(names,["WORKER","QUEEN"])
    for name,type in parameters:
        start = time.time()
        subprocess.run(" ./acotsp {} out.dat {} {} {} {} {} {}".format(
            name,
            type,
            args.max_iter,
            args.alpha,
            args.beta,
            args.evaporate,
            args.seed,
        ),shell=True)
        end = time.time()
        with open("out.dat") as f:
            solution = float(f.readline())

        print("{} {}: {:.4g} seconds, obtained {}, best known {}".format(type, name, end - start,solution,solutions_dict[name.split("/")[-1][:-4]]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all examples")
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2.,
    )
    parser.add_argument(
        "--evaporate",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    run_all(args)