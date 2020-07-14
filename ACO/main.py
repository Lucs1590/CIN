from time import time
from random import uniform, seed, random

import tsplib95
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from functools import reduce
# import operator


class Auxiliary(object):
    def read_file(self, path):
        _file = tsplib95.load(path)
        return pd.DataFrame.from_dict(_file.node_coords, orient='index')

    def define_seed(self, _seed=time()):
        print("Seed: ", _seed)
        seed(_seed)


class ACO(object):
    def __init__(self):
        self.aux = Auxiliary()

    def run_ACO(self, max_it, num_ants, pheromone):
        ...


def main():
    aux = Auxiliary()
    aco = ACO()

    aux.define_seed()
    start_time = time()
    cities = aux.read_file("ACO/berlin52.tsp")

    end_time = 0

    exec_time = end_time - start_time
    print("Tempo: ", exec_time)


if __name__ == "__main__":
    main()
