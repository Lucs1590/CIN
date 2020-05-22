from aux import AuxiliaryClass
from perceptron import Perceptron


def main():
    aux = AuxiliaryClass()
    perceptron = Perceptron()
    execute_iris(aux, perceptron)
    execute_wine(aux, perceptron)


def execute_iris(aux, perceptron):
    iris_data = aux.read_datas(
        "/home/brito/Documentos/Mestrado/CIN/algoritmos/Neural/iris/iris.data")
    print(iris_data)


def execute_wine(aux, perceptron):
    wine_data = aux.read_datas(
        "/home/brito/Documentos/Mestrado/CIN/algoritmos/Neural/iris/iris.data")
    print(wine_data)


if __name__ == "__main__":
    main()
