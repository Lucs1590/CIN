import numpy as np
import matplotlib.pyplot as plt
from random import uniform, seed
from time import time


class AuxiliaryClass(object):
    def __init__(self):
        ...

    def generate_population(self, indiv_number, dimensions):
        populations = []
        i = 0
        while i < dimensions:
            population = []

            for individual in range(indiv_number):
                population.append(uniform(-5, 5))

            populations.append(population)
            i += 1

        return populations

    def define_seed(self, _seed=time()):
        print("Seed: ", _seed)
        seed(_seed)


class PSOClass(object):
    def __init__(self):
        self.aux = AuxiliaryClass()

    def runPSO(self, flocks, max_it, AC1, AC2, v_min, v_max, dimensions):
        x = 0
        it = 0
        flocks = self.aux.generate_population(flocks, dimensions)
        speeds = self.generate_speed(flocks, v_min, v_max)

        while it < max_it:

            i = 0
            while i < len(flocks):
                curr_particle = flocks[i]
                best_optimal_indv = 0  # definir o melhor valor de todos que a particula já teve
                curr_optimal_indv = 0  # definir o melhor valor até hoje

                # colocar os valores comparados na função de aptidão
                if aux.fx(best_optimal_indv) > aux.fx(curr_optimal_indv):
                    curr_optimal_indv = best_optimal_indv

                neighbors = get_neighbors(curr_particle, flocks)
                best_neighbor = get_best_neightbor(neighbors)

                for neighbor in neighbors:
                    optimal_of_neighbor = 0  # aplicar função de apitidão aqui onde ta 0
                    if aux.fx(optimal_of_neighbor) > aux.fx(best_neighbor):
                        best_neighbor = optimal_of_neighbor
                i += 1

                speeds = self.update_speeds()

            it += 1

    def generate_speed(self, flock, v_min, v_max):
        flocks_speeds = []
        idx_1 = 0

        while idx_1 < len(flock):
            speeds = []
            idx_2 = 0

            while idx_2 < len(flock[idx_1]):
                speeds.append(uniform(v_min, v_max))
                idx_2 += 1
            flocks_speeds.append(speeds)
            idx_1 += 1

        return flocks_speeds

    def get_neighbors(self, curr_particle, flock):
        ...

    def get_best_neightbor(self, neighbors):
        ...

    def update_speeds(self):
        ...


class ACOClass(object):
    def __init__(self):
        self.aux = AuxiliaryClass()

    def run_ACO(self):
        ...


def main():
    aux = AuxiliaryClass()
    pso = PSOClass()
    (n_indiv, max_it, AC1, AC2, v_min, v_max, dimensions) = (
        8, 500, 2.05, 2.05, 64, 96, 2)
    aux.define_seed()

    start_time = time()
    pso.runPSO(n_indiv, max_it, AC1, AC2, v_min, v_max, dimensions)
    print("Tempo: ", time() - start_time)


if __name__ == "__main__":
    main()
