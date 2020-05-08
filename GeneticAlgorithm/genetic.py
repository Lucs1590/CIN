from random import randint, seed, random, sample
from time import time
import operator
from functools import reduce


class GeneticAlgorithm(object):

    def __init__(self):
        self.gc = GenericClass()

    def generate_population(self, indiv_number, _seed=time()):
        print("Seed: ", _seed)
        seed(_seed)
        population = []

        for individual in range(indiv_number):
            population.append(self.gc.to_bin(randint(0, 3951)))

        return population

    def select(self, aptitude, population, needle_points):
        individuals_score = self.gc.define_individuals_score(aptitude)
        population_scores = self.gc.group_sort_population_score(
            population, individuals_score)
        roulette = self.gc.define_roulette_positions_values(population_scores)
        selected_individuals = self.gc.select_individuals(
            roulette, needle_points)
        print("Selected Individuals: ", selected_individuals)
        return selected_individuals

    def reproduce(self, stallions):
        population_pair = self.gc.pair_stallions(stallions)
        cross_chance = self.gc.generate_random_chance(population_pair)

    def vary(self, parameter_list):
        pass

    def evaluate(self, parameter_list):
        pass

    def run_genetic_algorithm(self, goal):
        population = self.generate_population(8)
        print("Population: ", population)
        hamming_distance = self.gc.calculate_hamming(population, goal)
        # print("Hamming Distance: ", hamming_distance)
        aptitude = self.gc.calculate_aptitude(hamming_distance, 14)
        # print("Aptitude: ", aptitude)
        roulette_needles = self.gc.define_needle_points(8)
        stallions = self.select(aptitude, population, roulette_needles)
        new_population = self.reproduce(stallions)


class GenericClass(object):
    def to_bin(self, int_number):
        return format(int_number, '#014b')

    def to_number(self, bin_number):
        return int(bin_number, 2)

    def calculate_hamming(self, population, goal):
        hamming_distance = []
        for individual in population:
            hamming_distance.append(self.hamming(goal, individual))
        return hamming_distance

    def hamming(self, seq_1, seq_2):
        index = 0
        min_length = min(len(seq_1), len(seq_2))
        max_length = max(len(seq_1), len(seq_2))
        for i in range(min_length):
            if seq_1[i] != seq_2[i]:
                index = index + 1
        index = index + (max_length - min_length)
        return index

    def calculate_aptitude(self, hamming_distance, indiv_length):
        # removing two because binary, in pyhton, start with 0b
        indiv_length = indiv_length - 2
        points = []
        for point in hamming_distance:
            points.append(indiv_length - point)
        return points

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

    def define_individuals_score(self, population):
        elements_value = []
        elements_sum = reduce(operator.add, population)
        for individual in population:
            elements_value.append(individual/elements_sum)
        return elements_value

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
                if needles < individual[1] and needles >= individual[0]:
                    selecteds.append(individual[2])
        return selecteds

    def pair_stallions(self, population):
        population_pair = []
        i = 2
        while i < len(population):
            population_pair.append(population[i-2:i])
            i += 2
        return population_pair

    def generate_random_chance(self, population_pair):
        chances = []
        for chance in population_pair:
            chances.append(random())
        return chances


def main():
    genetic = GeneticAlgorithm()
    gc = GenericClass()

    genetic.run_genetic_algorithm('0b111101101111')


if __name__ == "__main__":
    main()
