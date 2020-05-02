from random import randint, seed
from time import time


class GeneticAlgorithm(object):
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

    def generate_population(self, indiv_number, _seed=time()):
        print(_seed)
        seed(_seed)
        population = []

        for individual in range(indiv_number):
            population.append(self.to_bin(randint(0, 3951)))

        return population


def main():
    genetic = GeneticAlgorithm()
    gc = GenericClass()

    print(*gc.generate_population(8), sep='\n')
    # genetic.run_genetic_algorithm()


if __name__ == "__main__":
    main()
