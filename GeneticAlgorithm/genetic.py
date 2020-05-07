from random import randint, seed, random
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
        population_numb = []
        roulette_needles = self.gc.define_needle_points(needle_points)
        # print("Needles Points: ", roulette_needles)
        individuals_point = self.gc.define_individuals_points(aptitude)
        print("Individuals Points: ", individuals_point)
        rolette = self.gc.define_rolette_positions_values(
            individuals_point, population)
        print("Rolette: ", rolette)

    def reproduce(self, parameter_list):
        pass

    def vary(self, parameter_list):
        pass

    def evaluate(self, parameter_list):
        pass

    def run_genetic_algorithm(self, goal):
        population = self.generate_population(8)
        print("Population: ", population)
        hamming_distance = self.gc.calculate_hamming(population, goal)
        print("Hamming Distance: ", hamming_distance)
        aptitude = self.gc.calculate_aptitude(hamming_distance, 14)
        print("Aptitude: ", aptitude)
        stallions = self.select(aptitude, population, 4)


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
        needle = 0
        random_number = random()
        while needle <= 1:
            needles.append(round(needle, 2))
            needle = space
            space += space
        for needle in needles:
            random_needles.append(needle*random_number*360)
        return random_needles

    def define_individuals_points(self, population):
        elements_value = []
        elements_sum = reduce(operator.add, population)
        for individual in population:
            elements_value.append(individual/elements_sum)
        return elements_value

    def define_rolette_positions_values(self, points, population):
        i = 0
        prev_value = 0
        rolette = []
        while i < len(population):
            degrees = prev_value + (points[i] * 360)
            rolette.append([prev_value, degrees, population[i]])
            prev_value = float(rolette[-1][1])
            i += 1
        return rolette


def main():
    genetic = GeneticAlgorithm()
    gc = GenericClass()

    genetic.run_genetic_algorithm('0b111101101111')
    # genetic.run_genetic_algorithm()


if __name__ == "__main__":
    main()
