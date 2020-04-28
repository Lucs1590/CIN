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
        return bin(int_number)

    def to_number(self, bin_number):
        return int(bin_number, 2)


def main():
    genetic = GeneticAlgorithm()
    gc = GenericClass()

    genetic.run_genetic_algorithm()


if __name__ == "__main__":
    main()
