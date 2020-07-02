import numpy as np
import matplotlib.pyplot as plt
from random import uniform, seed, random
from time import time
from functools import reduce
import operator
import pandas as pd


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

    def plot_poits(self,
                   data_1,
                   data_2,
                   title="Custo e Resultado",
                   label_1="Custo",
                   label_2="Soluções"
                   ):
        df = pd.DataFrame({"custo": data_1, "resultado": data_2})
        plt.subplot(211)
        plt.plot("custo", data=df, color="red")
        plt.title(title)
        plt.ylabel(label_1)

        plt.subplot(212)
        plt.plot("resultado", data=df, color="green")
        plt.xlabel("Interações")
        plt.ylabel(label_2)
        plt.show()


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
        min_aptitudes = []
        avg_aptitudes = []

        flocks = self.aux.generate_population(particles, dimensions)
        speeds = self.generate_speed(flocks, v_min, v_max)

        while it < max_it:
            i = 0

            aptitudes = self.calculate_aptitudes(flocks, dimensions)

            while i < len(flocks[0]):
                best_aptitude_indv[i] = self.get_best_indv_apt(
                    best_aptitude_indv, aptitudes, i)

                neighbors = self.get_neighbors(flocks, i)
                best_neighbor[i] = flocks[0].index(
                    self.get_best_neightbor(neighbors)[0])

                speeds = self.update_speeds(
                    v_min, v_max, speeds, flocks, best_aptitude_indv[best_neighbor[i]], best_aptitude_indv[i], AC1, AC2, i)
                flocks = self.update_movement(flocks, speeds, i)

                i += 1

            flocks_history.append(flocks)
            speeds_history.append(speeds)
            avg_aptitudes.append(
                (reduce(operator.add, aptitudes)/len(aptitudes)))
            min_aptitudes.append(min(aptitudes))
            it += 1

        finished_time = time()
        best_values = flocks_history[min_aptitudes.index(min(min_aptitudes))]

        self.aux.plot_poits(min_aptitudes,
                            avg_aptitudes,
                            "Aptidão Min. e Aptidão Média",
                            "Aptidão Min.",
                            "Aptidão Média"
                            )

        return min(best_values[0]), min(best_values[1]), round(self.aux.func_cost(min(best_values[0]), min(best_values[1])), 2), finished_time

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
        return [neighbors[0][0], neighbors[1][0]] \
            if self.aux.func_cost(neighbors[0][0], neighbors[0][1]) < self.aux.func_cost(neighbors[1][0], neighbors[1][1])\
            else [neighbors[0][1], neighbors[1][1]]

    def update_speeds(self, v_min, v_max, speeds, flocks, best_neighbor, best_apt, AC1, AC2, index):
        speeds[0][index] = (speeds[0][index]) + (random() * AC1) * (best_apt - flocks[0][index]) + (
            random() * AC2) * (best_neighbor - flocks[0][index])
        speeds[1][index] = speeds[1][index] + (random() * AC1) * (best_apt - flocks[1][index]) + (
            random() * AC2) * (best_neighbor - flocks[1][index])

        speeds[0][index] = self.speed_limit(speeds[0], v_min, v_max, index)
        speeds[1][index] = self.speed_limit(speeds[1], v_min, v_max, index)
        return speeds

    def speed_limit(self, speeds, v_min, v_max, index):
        if speeds[index] < v_min:
            return v_min
        elif speeds[index] > v_max:
            return v_max
        else:
            return speeds[index]

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
    (best_x, best_y, cost, end_time) = pso.runPSO(
        n_indiv, max_it, AC1, AC2, v_min, v_max, dimensions)

    exec_time = end_time - start_time
    print("Tempo: ", exec_time)
    print("Individuo (X): ", best_x)
    print("Individuo (Y): ", best_y)
    print("f(x,y): ", cost)


if __name__ == "__main__":
    main()
