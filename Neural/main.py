from aux import AuxiliaryClass
from perceptron import Perceptron
from time import time


def main():
    seed = time()
    aux = AuxiliaryClass()
    perceptron = Perceptron()
    execute_iris(aux, perceptron, seed)
    # execute_wine(aux, perceptron, seed)


def execute_iris(aux, perceptron, seed):
    iris_data = aux.read_datas("Neural/iris/iris.data")
    iris_data = aux.define_dataframe_column(
        iris_data, ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    (train_dataset, test_dataset, validation_dataset) = aux.divide_dataset(
        iris_data, 0.7, 0.2, 0.1, seed)
    print(train_dataset)


def execute_wine(aux, perceptron, seed):
    wine_data = aux.read_datas("Neural/wine/wine.data")
    print(wine_data)


if __name__ == "__main__":
    main()
