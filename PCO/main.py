import numpy as np
import matplotlib.pyplot as plt
from random import uniform, seed
from time import time


class AuxiliaryClass(object):
    def __init__(self):
        ...

    def define_seed(self, _seed=time()):
        print("Seed: ", _seed)
        seed(_seed)

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

    def func_cost(self, x, y):
        return (1 - x)**2 + 100*(y - x**2)**2


class PSOClass(object):
    def __init__(self):
        self.aux = AuxiliaryClass()

    def runPSO(self, particles, max_it, AC1, AC2, v_min, v_max, dimensions):
        x = 0
        it = 0
        best_optimals = []
        best_aptitude_indv = [0] * particles

        flocks = self.aux.generate_population(particles, dimensions)
        speeds = self.generate_speed(flocks, v_min, v_max)
        aptitudes = self.calculate_aptitudes(flocks, dimensions)

        while it < max_it:

            i = 0
            while i < len(flocks[0]):
                best_aptitude_indv[i] = aptitudes[i] if \
                    aptitudes[i] < best_aptitude_indv[i] else \
                    best_aptitude_indv[i]

                neighbors = get_neighbors(flocks, i)
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

    def calculate_aptitudes(self, flocks, dimensions):
        aptitudes = []
        i = 0

        while i < len(flocks[0]):
            aptitudes.append(self.aux.func_cost(flocks[0][i], flocks[1][i]))
            i += 1
        return aptitudes

    def get_neighbors(self, flock, index):
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
