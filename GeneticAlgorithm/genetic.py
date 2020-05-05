from random import randint, seed
from time import time


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

    def select(self, parameter_list):
        pass

    def reproduce(self, parameter_list):
        pass

    def vary(self, parameter_list):
        pass

    def evaluate(self, parameter_list):
        pass

    def run_genetic_algorithm(self, parameter_list):
        pass


class GenericClass(object):
    def to_bin(self, int_number):
        return format(int_number, '#014b')

    def to_number(self, bin_number):
        return int(bin_number, 2)

    def hamming(self, seq_1, seq_2):
        index = 0
        min_length = min(len(seq_1), len(seq_2))
        max_length = max(len(seq_1), len(seq_2))
        for i in range(min_length):
            if seq_1[i] != seq_2[i]:
                index = index + 1
        index = index + (max_length - min_length)
        return index


def main():
    genetic = GeneticAlgorithm()
    gc = GenericClass()

    genetic.run_genetic_algorithm()
    # genetic.run_genetic_algorithm()


if __name__ == "__main__":
    main()
