from aux import AuxiliaryClass
from perceptron import Perceptron
from time import time


def main():
    aux = AuxiliaryClass()
    perceptron = Perceptron()

    seed = time()
    max_it = 5000
    print("Seed: ", seed)

    execute_iris(aux, perceptron, seed, max_it)
    execute_wine(aux, perceptron, seed, max_it)


def execute_iris(aux, perceptron, seed, max_it):
    iris_data = aux.read_datas("Neural/iris/iris.data")
    iris_data = aux.define_dataframe_column(
        iris_data, ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    (train_dataset, test_dataset, validation_dataset) = aux.divide_dataset(
        iris_data, 0.7, 0.15, 0.15, seed)
    perceptron.run_perceptron(
        train_dataset, test_dataset, validation_dataset, max_it)


def execute_wine(aux, perceptron, seed, max_it):
    wine_data = aux.read_datas("Neural/wine/wine.data")
    print(wine_data)


if __name__ == "__main__":
    main()
