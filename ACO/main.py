from time import time
from random import uniform, seed, random

import tsplib95

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
            tingling.append([chosen])
            cities = cities.drop(chosen.index)
            idx += 1
        return tingling

    def get_euclidean_distance_and_comb(self, ant, next_move_coord):
        curr_ant_coord = (ant[0], ant[1])
        next_coord = (next_move_coord[0], next_move_coord[1])
        combination = str(curr_ant_coord[0]) + "x" + str(
            curr_ant_coord[1]) + "x" + str(next_coord[0]) + "x" + str(next_coord[1])
        return combination, tsplib95.distances.euclidean(curr_ant_coord, next_coord)

    def divide_by_total_sum(self, list_number):
        np_list_number = np.array(list_number)
        list_number /= np.sum(np_list_number)
        return list(list_number)

    def upgrade_total_distance(self, total_distance, total_distance_it):
        idx = 0
        while idx < len(total_distance):
            total_distance[idx] = total_distance[idx] + total_distance_it[idx]
            idx += 1
        return total_distance


class ACO(object):
    def __init__(self):
        self.aux = Auxiliary()

    def run_ACO(self, max_it, alpha, beta, evaporation, num_ants, cities, Q, initial_pheromone, elitist):
        it = 0
        best_solution = []

        self.problem = self.aux.read_original_file("ACO/berlin52.tsp")
        self.pheromone_way = {}

        pheromone_ways = initial_pheromone

        while it < max_it:
            idx = 0

            ants_distance = [0] * num_ants
            tingling = self.aux.define_tingling(num_ants, cities)

            while idx < len(cities)-1:
                best_solution, total_distance_it, pheromone_ways = self.choose_next_city(
                    cities, tingling, pheromone_ways, alpha, beta)

                tingling = self.move_ants(tingling, best_solution)

                ants_distance = self.aux.upgrade_total_distance(
                    ants_distance, total_distance_it)

                idx += 1

            best_solution = self.compare_solutions()
            self.upgrade_ways()

            it += 1

        end_time = time()
        best_sequence = min(min_ways)
        best_it = min_ways.index(best_sequence)

        return end_time, best_sequence, best_it

    def choose_next_city(self, cities, ants, pheromone_ways, alpha, beta):
        idx = 0
        moves = []
        total_distance = []

        while idx < len(ants):
            curr_ant = ants[idx]

            probability = self.get_probability_move(
                curr_ant, pheromone_ways, alpha, beta)

            probability = self.aux.divide_by_total_sum(probability)
            next_move_idx = probability.index(max(probability)) + 1

            moves.append(cities.loc[next_move_idx, ])

            total_distance.append(self.aux.get_euclidean_distance_and_comb(
                curr_ant[-1].values.tolist()[0], self.problem.node_coords[next_move_idx])[1])

            idx += 1
        return moves, total_distance, self.pheromone_way

    def get_probability_move(self, ant, pheromone_ways, alpha, beta):
        idx = 0
        curr_ant_probabilities = []
        idx_list = list(map(lambda x: x.index.values.tolist()[0], ant))

        ant = ant[-1].values.tolist()[0]

        # nesse while vai ter algo das cidades ja visitadas
        while idx < len(list(self.problem.get_nodes())):
            next_move_idx = list(self.problem.get_edges())[idx][1]
            next_move_coord = self.problem.node_coords[next_move_idx]

            if ant != next_move_coord and next_move_idx not in idx_list:
                (combination, distance) = self.aux.get_euclidean_distance_and_comb(
                    ant, next_move_coord)

                # se não tiver a chave com (and,next_move) em pheromone_ways adicionar
                self.pheromone_way_validation(pheromone_ways, combination)

                visibility = 1/distance
                probability = (
                    self.pheromone_way[combination] ** alpha) * (visibility**beta)
                curr_ant_probabilities.append(probability)
            else:
                curr_ant_probabilities.append(0)

            idx += 1

        return curr_ant_probabilities

    def pheromone_way_validation(self, pheromone, combination):
        if isinstance(pheromone, float):
            self.pheromone_way[combination] = pheromone
        elif combination not in pheromone.keys():
            self.pheromone_way[combination] = pheromone
        else:
            self.pheromone_way = pheromone

    def move_ants(self, tingling, best_solution):
        idx = 0
        while idx < len(tingling):
            tingling[idx].append(best_solution[idx].to_frame().T)
            idx += 1
        return tingling


def main():
    aux = Auxiliary()
    aco = ACO()

    cities = aux.read_file("ACO/berlin52.tsp")
    # aux.plot_cities(cities)

    aux.define_seed()
    start_time = time()

    (end_time, best_sequence, it) = aco.run_ACO(
        3, 1, 5, 0.5, len(cities), cities, 100, 0.000_001, 5)

    exec_time = end_time - start_time

    aux.plot_cities(best_sequence)
    print("Tempo: ", exec_time)
    print("Melhor caminho: ", best_sequence)
    print("Nº da Interação:", it)


if __name__ == "__main__":
    main()
