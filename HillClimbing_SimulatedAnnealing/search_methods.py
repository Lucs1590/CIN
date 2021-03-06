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
import struct
from codecs import decode


class GenericClass(object):
    def bin_to_float(self, b):
        bf = self.int_to_bytes(int(b, 2), 8)
        return struct.unpack('>d', bf)[0]

    def int_to_bytes(self, n, length):
        return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]

    def float_to_bin(self, value):
        [d] = struct.unpack(">Q", struct.pack(">d", value))
        return '{:064b}'.format(d)

    def get_random_point(self, _seed):
        seed(_seed)
        return round(random(), 2)

    def disturb_point(self, point):
        return float(point + np.random.normal(0, 0.001, 1))

    def func_g_x(self, x):
        return 2 ** (-2 * ((((x-0.1) / 0.9)) ** 2)) * ((math.sin(5*math.pi*x)) ** 6)

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

    def format_table(self,
                     hc_result, hc_cost, hc_time,
                     sa_result, sa_cost, sa_time,
                     gen_result, gen_cost, gen_time
                     ):
        formated_table = [
            ["Método", "Solução(x)", "Custo", "Tempo"],
            ["Hill Climbing", hc_result, hc_cost, hc_time],
            ["Simulated Annealing", sa_result, sa_cost, sa_time],
            ["Genetic Algorithm", gen_result, gen_cost, gen_time]
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

    def pair_stallions(self, population):
        population_pair = []
        i = 2
        while i <= len(population):
            population_pair.append(population[(i-2):i])
            i += 2
        return population_pair

    def generate_random_chance(self, population_pair):
        chances = []
        for chance in population_pair:
            chances.append(random())
        return chances


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

    def run_genetic_algorithm(self, seed, indiv_number, max_it):
        it = 1
        it_repeat = 0
        min_aptitudes = []
        aptitudes_avg = []
        population = self.generate_population(indiv_number, seed)

        while it < max_it:
            if (it > 2):
                it_repeat += 1 if (aptitudes_avg[-1]
                                   == aptitudes_avg[-2]) else 0
            if (it_repeat == round(max_it/3)):
                print("(GN) Motivo de parada: Sem melhorias!")
                break

            aptitudes = self.calculate_aptitudes(population)
            i = next(('break' for elem in aptitudes if elem > 1), None)
            if i:
                print("(GN) Motivo de parada: Aptidão superou o limite!")
                eval(i)
            aptitudes_avg.append(
                (reduce(operator.add, aptitudes)/len(aptitudes)))
            min_aptitudes.append(min(aptitudes))
            roulette_needles = self.gc.define_needle_points(indiv_number)
            stallions = self.select(aptitudes, population, roulette_needles)
            new_generation = self.reproduce(stallions)
            mutated_new_generation = self.mutate(new_generation)

            it += 1
            population = mutated_new_generation

        print("(GN) Motivo de parada: Número máximo de interações atingido!")
        finished_time = time()
        self.gc.plot_poits(min_aptitudes,
                           aptitudes_avg,
                           "Aptidão Min. e Aptidão Média",
                           "Aptidão Min.",
                           "Aptidão Média"
                           )
        return max(population), aptitudes_avg[-1], finished_time

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

    def reproduce(self, stallions):
        population_pair = self.gc.pair_stallions(stallions)
        cross_chances = self.gc.generate_random_chance(population_pair)
        new_generation = self.cross_over(
            population_pair, cross_chances, randint(1, 63))
        return new_generation

    def cross_over(self, population_pair, cross_chances, crop, Pc=0.6):
        self.new_generation = []
        i = 0
        while i < len(cross_chances):
            if cross_chances[i] <= Pc:
                individuals_pair = self.make_cross_over(
                    population_pair[i], crop)
                self.make_new_generation(individuals_pair)
            else:
                self.make_new_generation(population_pair[i])
            i += 1
        return self.new_generation

    def make_cross_over(self, population_pair, crop):
        cropped_start = []
        cropped_end = []
        for individual in population_pair:
            individual = self.gc.float_to_bin(individual)
            cropped_start.append(individual[2:crop])
            cropped_end.append(individual[crop:])
        return [
            self.gc.bin_to_float(cropped_start[0] + cropped_end[1]),
            self.gc.bin_to_float(cropped_start[1] + cropped_end[0])
        ]

    def make_new_generation(self, population_pair):
        for individual in population_pair:
            self.new_generation.append(individual)

    def mutate(self, population):
        mutated_list = []
        for individual in population:
            random_chance = random()
            if random_chance <= 0.02:
                individual = self.gc.disturb_point(individual)
            mutated_list.append(individual)
        return mutated_list


def main():
    seed = time()

    max_it = 500
    T = 500
    min_value = 0.9

    hc = HillClimbing()
    sa = SimulatedAnnealing()
    gen = GeneticAlgorithm()

    print("Semente: ", seed)

    start_hc_time = time()
    (hc_best_result, hc_cost, end_hc_time) = hc.run_hill_climbing(
        max_it, min_value, seed)
    hc_time = end_hc_time - start_hc_time

    start_sa_time = time()
    (sa_best_result, sa_cost, end_sa_time) = sa.run_simulated_annealing(T, seed)
    sa_time = end_sa_time - start_sa_time

    start_genetic_time = time()
    (gen_best_result, gen_cost, end_genetic_time) = gen.run_genetic_algorithm(
        seed, 8, max_it)
    gen_time = end_genetic_time - start_genetic_time

    GenericClass.format_table(GenericClass,
                              hc_best_result, hc_cost, hc_time,
                              sa_best_result, sa_cost, sa_time,
                              gen_best_result, gen_cost, gen_time
                              )


if __name__ == "__main__":
    main()

# Fazer as respostas no readme
