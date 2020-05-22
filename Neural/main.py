from aux import AuxiliaryClass
from perceptron import Perceptron


def main():
    aux = AuxiliaryClass()

    iris_data = aux.read_datas(
        "/home/brito/Documentos/Mestrado/CIN/algoritmos/Neural/iris/iris.data")
    print(iris_data)

    wine_data = aux.read_datas(
        "/home/brito/Documentos/Mestrado/CIN/algoritmos/Neural/iris/iris.data")
    print(wine_data)

    perceptron = Perceptron()


if __name__ == "__main__":
    main()
