from time import time
from random import uniform, seed, random

import tsplib95
import pandas as pd
import matplotlib.pyplot as plt

# import numpy as np

# from functools import reduce
# import operator


class Auxiliary(object):
    def read_file(self, path):
        _file = tsplib95.load(path)
        return pd.DataFrame.from_dict(_file.node_coords, orient='index')

    def define_seed(self, _seed=time()):
        print("Seed: ", _seed)
        seed(_seed)

    def plot_cities(self, cities):
        cities.columns = ["width", "height"]
        plt.scatter(cities.width, cities.height)
        plt.title("Initial Graph")
        plt.show()


class ACO(object):
    def __init__(self):
        self.aux = Auxiliary()

    def run_ACO(self, max_it, num_ants, pheromone):
        it = 0
        min_ways = []

        tingling = self.aux.define_tingling(num_ants)
        best_solution = self.define_best_solution(tingling)

        while it < max_it:
            idx = 0

            while idx < len(tingling):
                # prev_best_solution = curr_best_solution
                curr_best_solution = self.compare_solutions(
                    curr_best_solution, prev_best_solution)
                idx += 1

            best_solution = self.compare_solutions(
                best_solution, curr_best_solution)
            self.upgrade_ways()

            it += 1

        end_time = time()
        best_sequence = min(min_ways)
        best_it = min_ways.index(best_sequence)

        return end_time, best_sequence, best_it


def main():
    aux = Auxiliary()
    aco = ACO()

    cities = aux.read_file("ACO/berlin52.tsp")
    aux.plot_cities(cities)

    aux.define_seed()
    start_time = time()

    (end_time, best_sequence, it) = aco.run_ACO(500, 50, 0.4)

    exec_time = end_time - start_time

    aux.plot_cities(best_sequence)
    print("Tempo: ", exec_time)
    print("Melhor caminho: ", best_sequence)
    print("Nº da Interação:", it)


if __name__ == "__main__":
    main()
