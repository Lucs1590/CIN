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
        _file = self.read_original_file(path)
        return pd.DataFrame.from_dict(_file.node_coords, orient='index')

    def read_original_file(self, path):
        return tsplib95.load(path)

    def define_seed(self, _seed=time()):
        print("Seed: ", _seed)
        seed(_seed)

    def plot_cities(self, cities):
        cities.columns = ["width", "height"]
        plt.scatter(cities.width, cities.height)
        plt.title("Initial Graph")
        plt.show()

    def define_tingling(self, num_ants, cities):
        idx = 0
        tingling = []
        while idx < num_ants:
            chosen = cities.sample(1)
            tingling.append(chosen.values)
            cities = cities.drop(chosen.index)
            idx += 1
        return tingling

    def get_euclidean_distance(self, ant, next_move_coord):
        curr_ant_coord = (ant[0], ant[1])
        next_coord = (next_move_coord[0], next_move_coord[1])
        return tsplib95.distances.euclidean(curr_ant_coord, next_coord)


class ACO(object):
    def __init__(self):
        self.aux = Auxiliary()

    def run_ACO(self, max_it, num_ants, pheromone, cities):
        it = 0
        best_solution = []
        min_ways = []

        self.problem = self.aux.read_original_file("ACO/berlin52.tsp")

        while it < max_it:
            idx = 0

            ants_distance = [0] * num_ants
            tingling = self.aux.define_tingling(num_ants, cities)

            while idx < len(cities)-1:
                best_solution = self.move_ants(tingling)
                idx += 1

            best_solution = self.compare_solutions()
            self.upgrade_ways()

            it += 1

        end_time = time()
        best_sequence = min(min_ways)
        best_it = min_ways.index(best_sequence)

        return end_time, best_sequence, best_it

    def move_ants(self, ants):
        idx = 0

        while idx < len(ants):
            curr_ant = ants[idx]
            probability = self.get_probability_move(curr_ant)

            idx += 1

    def get_probability_move(self, ant):
        idx = 0
        curr_ant_probabilities = []
        ant = ant.tolist()[0]

        while idx < len(list(self.problem.get_nodes())):
            next_move_idx = list(self.problem.get_edges())[idx][1]
            next_move_coord = self.problem.node_coords[next_move_idx]
            if ant != next_move_coord:
                distance = self.aux.get_euclidean_distance(
                    ant, next_move_coord)

                visibility = 1/distance
                curr_ant_probabilities = ...

            idx += 1

        return curr_ant_probabilities


def main():
    aux = Auxiliary()
    aco = ACO()

    cities = aux.read_file("ACO/berlin52.tsp")
    # aux.plot_cities(cities)

    aux.define_seed()
    start_time = time()

    (end_time, best_sequence, it) = aco.run_ACO(500, len(cities), 0.4, cities)

    exec_time = end_time - start_time

    aux.plot_cities(best_sequence)
    print("Tempo: ", exec_time)
    print("Melhor caminho: ", best_sequence)
    print("Nº da Interação:", it)


if __name__ == "__main__":
    main()
