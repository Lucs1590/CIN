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
                population.append(uniform(0, 0.001))

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
        best_aptitude_indv = [0] * particles
        best_neighbor = [0] * particles
        flocks_history = []
        speeds_history = []
        aptitudes_history = []

        flocks = self.aux.generate_population(particles, dimensions)
        speeds = self.generate_speed(flocks, v_min, v_max)
        aptitudes = self.calculate_aptitudes(flocks, dimensions)

        while it < max_it:

            i = 0
            while i < len(flocks[0]):
                # ver a possibilidade de mudar de best_aptitude_indv para aptitudes
                best_aptitude_indv[i] = self.get_best_indv_apt(
                    best_aptitude_indv, aptitudes, i)

                neighbors = self.get_neighbors(flocks, i)
                best_neighbor[i] = flocks[0].index(
                    self.get_best_neightbor(neighbors)[0])

                speeds = self.update_speeds(
                    v_min, v_max, speeds, flocks, AC1, AC2, i)
                flocks = self.update_movement(flocks, speeds, i)

                i += 1

            flocks_history.append(flocks)
            speeds_history.append(speeds)
            # dar append em aptitude ou best_aptitude_indiv
            aptitudes_history.append()
            it += 1

        return flocks_history, speeds_history, aptitudes_history

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

    def get_best_indv_apt(self, best_aptitude, aptitudes, index):
        return aptitudes[index] if aptitudes[index] < best_aptitude[index] else best_aptitude[index]

    def get_neighbors(self, flock, index):
        neighbors_x = []
        neighbors_y = []

        if index == 0:
            neighbors_x.append(flock[0][-1])
            neighbors_y.append(flock[1][-1])

            neighbors_x.append(flock[0][index + 1])
            neighbors_y.append(flock[1][index + 1])
        elif (index == len(flock[0]) - 1):
            neighbors_x.append(flock[0][0])
            neighbors_y.append(flock[1][0])

            neighbors_x.append(flock[0][index - 1])
            neighbors_y.append(flock[1][index - 1])
        else:
            neighbors_x.append(flock[0][index - 1])
            neighbors_y.append(flock[1][index - 1])

            neighbors_x.append(flock[0][index + 1])
            neighbors_y.append(flock[1][index + 1])

        return [neighbors_x, neighbors_y]

    def get_best_neightbor(self, neighbors):
        return neighbors[0] \
            if self.aux.func_cost(neighbors[0][0], neighbors[0][1]) < self.aux.func_cost(neighbors[1][0], neighbors[1][1])\
            else neighbors[1]

    def update_speeds(self, v_min, v_max, speeds, flocks, AC1, AC2, index):
        # Fazer a atualização de velocidades speeds[0][index]
        while self.speed_limit(speeds, v_min, v_max, index):
            self.update_speeds(v_min, v_max, speeds, flocks, AC1, AC2, index)
        return speeds

    def speed_limit(self, speeds, v_min, v_max, index):
        return False \
            if v_min <= speeds[0][index] <= v_max and v_min <= speeds[1][index] <= v_max \
            else True

    def update_movement(self, flocks, speeds, index):
        flocks[0][index] = flocks[0][index] + speeds[0][index]
        flocks[1][index] = flocks[1][index] + speeds[1][index]
        return flocks


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
