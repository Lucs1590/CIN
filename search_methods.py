import random
from time import time
# 1587493971.024905 para <
# 1587503056.092915 para >
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from terminaltables import AsciiTable


class GenericClass(object):

    def get_random_point(self, seed):
        print("Semente: ", seed)
        random.seed(seed)
        return round(random.random(), 2)

    def disturb_point(self, point):
        return float(point + abs(np.random.normal(0, 0.001, 1)))

    def func_g_x(self, x):
        return (2**-2*(x-0.1/0.9)**2)*(math.sin(5*math.pi*x))**6

    def plot_poits(self, data_cost, data_result):
        df = pd.DataFrame({"custo": data_cost, "resultado": data_result})
        plt.subplot(211)
        plt.plot("custo", data=df, color="red")
        plt.title("Custo e Resultado")
        plt.ylabel("Custo")

        plt.subplot(212)
        plt.plot("resultado", data=df, color="green")
        plt.xlabel("Interações")
        plt.ylabel("Resultados")
        plt.show()

    def format_table(self, hc_result, hc_cust, sa_result, sa_cost):
        formated_table = [
            ["Método", "Resultado x", "Custo"],
            ["Hill Climbing", hc_result, hc_cust],
            ["Simulated Annealing", sa_result, sa_cost]
        ]
        table = AsciiTable(formated_table)
        print(table.table)


class HillClimbing(object):

    def __init__(self):
        self.gc = GenericClass()
        self.set_default_values()

    def set_default_values(self):
        self.it = 0
        self.it_repeat = 0
        self.cost = 0
        self.cost_list = []
        self.results_list = []

    def run_hill_climbing(self, max_it, min_value, seed):
        current_best_result = self.gc.get_random_point(seed)
        last_best_result = current_best_result

        (current_best_result, cost) = self.evaluate_point(
            current_best_result, last_best_result)

        self.cost_list.append(cost)
        self.results_list.append(current_best_result)

        while self.it < max_it:
            self.it_repeat += 1 if current_best_result == last_best_result else 0
            if (self.it_repeat == round(max_it/3)):
                print("(HC) Motivo de parada: Sem melhorias!")
                break
            last_best_result = current_best_result
            current_best_result = self.gc.disturb_point(current_best_result)
            (current_best_result, cost) = self.evaluate_point(
                current_best_result, last_best_result)

            self.cost_list.append(cost)
            self.results_list.append(current_best_result)

            self.it += 1
        print(
            "(HC) Motivo de parada: Número máximo de interações atingido!") if max_it == self.it else ...
        self.gc.plot_poits(self.cost_list, self.results_list)
        return current_best_result, cost

    def evaluate_point(self, best_result, last_best_result):
        current_cost = self.gc.func_g_x(best_result)
        last_cost = self.gc.func_g_x(last_best_result)
        return (best_result, current_cost) if last_cost < current_cost else (last_best_result, last_cost)


def simulated_annealing(max_it, temperature, seed):
    gc = GenericClass()

    it = 0
    it_repeat = 0
    cost = 0
    cost_list = []
    results_list = []

    current_best_result = gc.get_random_point(seed)
    last_best_result = current_best_result

    (current_best_result, cost) = gc.evaluate_point(
        current_best_result, last_best_result)

    cost_list.append(cost)
    results_list.append(current_best_result)

    while (it < max_it) or (temperature <= 0.1):
        it_repeat += 1 if current_best_result == last_best_result else 0
        if (it_repeat == round(max_it/3)):
            print("Sem melhorias!")
            break
        last_best_result = current_best_result
        current_best_result = gc.disturb_point(current_best_result)

        # a diferença do hill climb e do simulated estão exatamente aqui
        (current_best_result, cost) = gc.evaluate_point(
            current_best_result, last_best_result)

        cost_list.append(cost)
        results_list.append(current_best_result)

        it += 1
    print("Numero máximo de interações atingido!") if max_it == it else ...
    gc.plot_poits(cost_list, results_list)
    return current_best_result, cost


def main():
    seed = 1587493971.024905
    max_it = 500
    min_value = 0.9
    hc = HillClimbing()

    (hc_best_result, hc_cost) = hc.run_hill_climbing(max_it, min_value, seed)
    # simulated_annealing(max_it, 100, seed)
    GenericClass.format_table(
        GenericClass, hc_best_result, hc_cost, 0, 0)  # sa_best_result, sa_cost)

    # (sa_best_results, sa_cost) = simulated_annealing()


if __name__ == "__main__":
    main()

# Criar classe para simulated anealing e evaluate simulated
