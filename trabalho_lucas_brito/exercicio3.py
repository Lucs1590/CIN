from time import time
# 1587607351.1839364
# 1589853324.1593766
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from terminaltables import AsciiTable
from random import randint, random, seed, sample, choice, uniform
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

    def disturb_point(self, point):
        return float(point - np.random.normal(0, 0.001, 1))

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

        if len(roulette) < 8:
            random_element = choice(roulette)
            avg = (random_element[1] + random_element[0]) / 2
            roulette[roulette.index(random_element)] = [
                avg, random_element[1], random_element[2]]
            roulette.insert(roulette.index([avg, random_element[1], random_element[2]]), [
                            random_element[0], avg, random_element[2]])
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


class GeneticAlgorithm(object):
    def __init__(self):
        self.gc = GenericClass()
        self.new_generation = []

    def run_genetic_algorithm(self, seed, indiv_number, max_it):
        it = 1
        it_repeat = 0
        min_aptitudes = []
        aptitudes_avg = []
        population_x_hist = []
        population_y_hist = []
        population_x = self.generate_population(indiv_number, seed)
        population_y = self.generate_population(indiv_number, seed)

        while it < max_it:
            population_x_hist.append(population_x)
            population_y_hist.append(population_y)

            if (it > 2):
                it_repeat += 1 if (aptitudes_avg[-1]
                                   == aptitudes_avg[-2]) else 0
            if (it_repeat == round(max_it/3)):
                print("(GN) Motivo de parada: Sem melhorias!")
                break

            aptitudes = self.calculate_aptitudes(population_x, population_y)
            aptitudes_avg.append(
                (reduce(operator.add, aptitudes)/len(aptitudes)))
            min_aptitudes.append(min(aptitudes))

            i_x = next(
                ('break' for elem in population_x if elem > 5 or elem < -5), None)
            i_y = next(
                ('break' for elem in population_y if elem > 5 or elem < -5), None)
            if i_x or i_y:
                print("(GN) Motivo de parada: Individuo superou o limite!")
                break

            roulette_needles_x = self.gc.define_needle_points(indiv_number)
            stallions_x = self.select(
                aptitudes, population_x, roulette_needles_x)

            roulette_needles_y = self.gc.define_needle_points(indiv_number)
            stallions_y = self.select(
                aptitudes, population_y, roulette_needles_y)

            new_generation_x = self.reproduce(stallions_x)
            mutated_new_generation_x = self.mutate(new_generation_x)

            new_generation_y = self.reproduce(stallions_y)
            mutated_new_generation_y = self.mutate(new_generation_y)

            it += 1
            population_x = mutated_new_generation_x
            population_y = mutated_new_generation_y

        print("(GN) Motivo de parada: Número máximo de interações atingido!")
        finished_time = time()
        best_x = population_x_hist[min_aptitudes.index(min(min_aptitudes))]
        best_y = population_y_hist[min_aptitudes.index(min(min_aptitudes))]

        self.gc.plot_poits(min_aptitudes,
                           aptitudes_avg,
                           "Aptidão Min. e Aptidão Média",
                           "Aptidão Min.",
                           "Aptidão Média"
                           )
        return round(min(best_x), 2), round(min(best_y), 2), round(self.gc.func_cost(min(best_x), min(best_y)), 2), finished_time

    def generate_population(self, indiv_number, _seed):
        population = []
        for individual in range(indiv_number):
            population.append(round(uniform(-5, 5), 2))
        return population

    def calculate_aptitudes(self, population_x, population_y):
        aptitudes = []
        i = 0
        while i <= 5:
            aptitudes.append(self.gc.func_cost(
                population_x[i], population_y[i]))
            i += 1
        return aptitudes

    def select(self, aptitude, population, needle_points):
        population.sort(reverse=True)
        individuals_score = self.gc.define_individuals_score(aptitude)
        individuals_score.sort()
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
    gen = GeneticAlgorithm()

    print("Semente: ", seed)
    start_genetic_time = time()
    (best_result_x, best_result_y, cost, end_genetic_time) = gen.run_genetic_algorithm(
        seed, 8, max_it)
    gen_time = end_genetic_time - start_genetic_time
    print("Tempo: ", gen_time)
    print("Individuo X: ", best_result_x)
    print("Individuo Y: ", best_result_y)
    print("f(x,y): ", cost)


if __name__ == "__main__":
    main()
