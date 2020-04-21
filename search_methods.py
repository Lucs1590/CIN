import random
from time import time
# 1587493971.024905
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from terminaltables import AsciiTable


class GenericClass(object):

    def get_random_point(self, seed):
        print(seed)
        random.seed(seed)
        return round(random.random(), 2)

    def disturb_point(self, point):
        return float(point + abs(np.random.normal(0, 0.001, 1)))

    def evaluate_point(self, best_result, last_best_result):
        current_cost = self.func_g_x(best_result)
        last_cost = self.func_g_x(last_best_result)
        return (best_result, current_cost) if last_cost < current_cost else (last_best_result, last_cost)

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

    def format_table(self, args, kwargs):
        table = AsciiTable(args)
        print(table.table)


def hill_climbing(max_it, min_value):
    it = 0
    it_repeat = 0
    cost = 0
    cost_list = []
    results_list = []
    gc = GenericClass()
    current_best_result = gc.get_random_point(time())
    last_best_result = current_best_result

    (current_best_result, cost) = gc.evaluate_point(
        current_best_result, last_best_result)

    cost_list.append(cost)
    results_list.append(current_best_result)

    while it < max_it:
        """coment """
        it_repeat += 1 if current_best_result == last_best_result else 0
        if (it_repeat == round(max_it/3)):
            print("Sem melhorias!")
            break
        last_best_result = current_best_result
        current_best_result = gc.disturb_point(current_best_result)
        (current_best_result, cost) = gc.evaluate_point(
            current_best_result, last_best_result)

        cost_list.append(cost)
        results_list.append(current_best_result)

        it += 1
    print("Numero máximo de interações atingido!") if max_it == it else ...
    gc.plot_poits(cost_list, results_list)
    return current_best_result, cost


def main():
    (hc_best_result, hc_cost) = hill_climbing(500, 0.9)
    # GenericClass.format_table()


if __name__ == "__main__":
    main()
