from time import time
# 1587607351.1839364
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from terminaltables import AsciiTable
from random import randint, random, seed, sample
import operator
from functools import reduce


class GenericClass(object):

    def get_random_point(self, _seed):
        seed(_seed)
        return round(random(), 2)

    def disturb_point(self, point):
        return float(point + np.random.normal(0, 0.001, 1))

    def func_g_x(self, x):
        return 2 ** (-2 * ((((x-0.1) / 0.9)) ** 2)) * ((math.sin(5*math.pi*x)) ** 6)

    def plot_poits(self, data_cost, data_result):
        df = pd.DataFrame({"custo": data_cost, "resultado": data_result})
        plt.subplot(211)
        plt.plot("custo", data=df, color="red")
        plt.title("Custo e Resultado")
        plt.ylabel("Custo")

        plt.subplot(212)
        plt.plot("resultado", data=df, color="green")
        plt.xlabel("Interações")
        plt.ylabel("Resultados")
        plt.show()

    def format_table(self, hc_result, hc_cust, sa_result, sa_cost):
        formated_table = [
            ["Método", "Resultado x", "Custo"],
            ["Hill Climbing", hc_result, hc_cust],
            ["Simulated Annealing", sa_result, sa_cost]
        ]
        table = AsciiTable(formated_table)
        print(table.table)

    def set_default_values(self):
        return (0, 0, 0, [], [])

    def define_needle_points(self, needle_points):
        needles = []
        random_needles = []
        space = 1 / needle_points
        space_it = space
        needle = 0
        while needle <= 1:
            needles.append(round(needle, 2))
            needle = space_it
            space_it += space
        for needle in needles:
            random_needles.append(needle*360)
        return random_needles

    def define_individuals_score(self, aptitudes):
        scores_value = []
        elements_sum = reduce(operator.add, aptitudes)
        for score in aptitudes:
            scores_value.append(score/elements_sum)
        return scores_value

    def group_sort_population_score(self, scores, population):
        population_score = []
        i = 0
        while i < len(population):
            population_score.append([population[i], scores[i]])
            i += 1
        return sample(population_score, len(population_score))

    def define_roulette_positions_values(self, population_score):
        i = 0
        prev_value = 0
        roulette = []
        while i < len(population_score):
            degrees = prev_value + (population_score[i][0] * 360)
            roulette.append([prev_value, degrees, population_score[i][1]])
            prev_value = float(roulette[-1][1])
            i += 1
        return roulette

    def select_individuals(self, roulette, roulette_needles):
        selecteds = []
        for needles in roulette_needles:
            for individual in roulette:
                if needles > individual[0] and needles <= individual[1]:
                    selecteds.append(individual[2])
        return selecteds


class HillClimbing(object):

    def __init__(self):
        self.gc = GenericClass()
        (self.it, self.it_repeat, self.cost, self.cost_list,
         self.results_list) = self.gc.set_default_values()

    def run_hill_climbing(self, max_it, min_value, seed):
        current_best_result = self.gc.get_random_point(seed)
        last_best_result = current_best_result

        while self.it < max_it:
            self.it_repeat += 1 if current_best_result == last_best_result else 0
            if (self.it_repeat == round(max_it/3)):
                print("(HC) Motivo de parada: Sem melhorias!")
                break
            last_best_result = current_best_result
            current_best_result = self.gc.disturb_point(current_best_result)
            (current_best_result, cost) = self.evaluate_point(
                current_best_result, last_best_result)

            self.cost_list.append(cost)
            self.results_list.append(current_best_result)

            self.it += 1
        print(
            "(HC) Motivo de parada: Número máximo de interações atingido!") if max_it == self.it else ...
        finished_time = time()
        self.gc.plot_poits(self.cost_list, self.results_list)
        return current_best_result, cost, finished_time

    def evaluate_point(self, best_result, last_best_result):
        current_cost = self.gc.func_g_x(best_result)
        last_cost = self.gc.func_g_x(last_best_result)
        return (best_result, current_cost) if last_cost < current_cost else (last_best_result, last_cost)


class SimulatedAnnealing(object):

    def __init__(self):
        self.gc = GenericClass()
        (self.it, self.it_repeat, self.cost, self.cost_list,
         self.results_list) = self.gc.set_default_values()

    def run_simulated_annealing(self, temperature, seed):
        penalty = 100 / (temperature * 1.10)
        current_best_result = self.gc.get_random_point(seed)
        last_best_result = current_best_result

        while temperature >= 0.1:
            last_best_result = current_best_result
            current_best_result = self.gc.disturb_point(current_best_result)

            (current_best_result, cost) = self.evaluate_point(
                current_best_result, last_best_result, temperature, penalty)

            self.cost_list.append(cost)
            self.results_list.append(current_best_result)

            temperature = self.reduce_temperature(temperature)
            self.it += 1

        print("(SA) Motivo de parada: Temperatura foi zerada!")
        finished_time = time()
        self.gc.plot_poits(self.cost_list, self.results_list)
        return self.results_list[self.cost_list.index(max(self.cost_list))], max(self.cost_list), finished_time

    def evaluate_point(self, best_result, last_best_result, T, penalty):
        current_cost = self.gc.func_g_x(best_result)
        last_cost = self.gc.func_g_x(last_best_result)
        probability = math.e**(last_cost - current_cost / T) - penalty

        if last_cost < current_cost:
            return (best_result, current_cost)
        elif random() <= probability:
            return (best_result, current_cost)
        else:
            return (last_best_result, last_cost)

    def reduce_temperature(self, T):
        # Fazer com função euleriana
        return T - (T/70)


class GeneticAlgorithm(object):
    def __init__(self):
        self.gc = GenericClass()
        self.new_generation = []

    def run_genetic_algorithm(self, seed, indiv_number, it):
        max_it = 1
        min_aptitudes = []
        aptitudes_avg = []
        population = self.generate_population(indiv_number, seed)
        print("Population: ", population)

        while max_it <= it:
            aptitudes = self.calculate_aptitudes(population)
            roulette_needles = self.gc.define_needle_points(indiv_number)
            stallions = self.select(aptitudes, population, roulette_needles)

    def generate_population(self, indiv_number, _seed):
        population = []
        for individual in range(indiv_number):
            population.append(round(random(), 2))
        return population

    def calculate_aptitudes(self, population):
        aptitudes = []
        for individual in population:
            aptitudes.append(self.gc.func_g_x(individual))
        return aptitudes

    def select(self, aptitude, population, needle_points):
        individuals_score = self.gc.define_individuals_score(aptitude)
        population_scores = self.gc.group_sort_population_score(
            population, individuals_score)
        roulette = self.gc.define_roulette_positions_values(population_scores)
        selected_individuals = self.gc.select_individuals(
            roulette, needle_points)
        return selected_individuals


def main():
    seed = 1588449669.2848306

    max_it = 500
    T = 500
    min_value = 0.9

    hc = HillClimbing()
    sa = SimulatedAnnealing()
    genetic = GeneticAlgorithm()

    print("Semente: ", seed)

    start_hc_time = time()
    (hc_best_result, hc_cost, end_hc_time) = hc.run_hill_climbing(
        max_it, min_value, seed)
    hc_time = end_hc_time - start_hc_time

    start_sa_time = time()
    (sa_best_result, sa_cost, end_sa_time) = sa.run_simulated_annealing(T, seed)
    sa_time = end_sa_time - start_sa_time

    start_genetic_time = time()
    genetic.run_genetic_algorithm(seed, 8, max_it)
    genetic_time = time() - start_genetic_time

    GenericClass.format_table(
        GenericClass, hc_best_result, hc_cost, sa_best_result, sa_cost)
    print("Tempo HC: ", hc_time)
    print("Tempo SA: ", sa_time)


if __name__ == "__main__":
    main()

# Fazer as respostas no readme
