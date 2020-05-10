import random
from time import time
# 1587607351.1839364
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from terminaltables import AsciiTable


class GenericClass(object):

    def get_random_point(self, seed):
        random.seed(seed)
        return round(random.random(), 2)

    def disturb_point(self, point):
        return float(point + abs(np.random.normal(0, 0.001, 1)))

    def func_g_x(self, x):
        return 2 ** (-2 * ((((x-0.1) / 0.9)) ** 2)) * ((math.sin(5*math.pi*x)) ** 6)

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

    def set_default_values(self):
        return (0, 0, 0, [], [])


class HillClimbing(object):

    def __init__(self):
        self.gc = GenericClass()
        (self.it, self.it_repeat, self.cost, self.cost_list,
         self.results_list) = self.gc.set_default_values()

    def run_hill_climbing(self, max_it, min_value, seed):
        current_best_result = self.gc.get_random_point(seed)
        last_best_result = current_best_result

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
        finished_time = time()
        self.gc.plot_poits(self.cost_list, self.results_list)
        return current_best_result, cost, finished_time

    def evaluate_point(self, best_result, last_best_result):
        current_cost = self.gc.func_g_x(best_result)
        last_cost = self.gc.func_g_x(last_best_result)
        return (best_result, current_cost) if last_cost < current_cost else (last_best_result, last_cost)


class SimulatedAnnealing(object):

    def __init__(self):
        self.gc = GenericClass()
        (self.it, self.it_repeat, self.cost, self.cost_list,
         self.results_list) = self.gc.set_default_values()

    def run_simulated_annealing(self, temperature, seed):
        penalty = it = 100 / (temperature * 1.10)
        current_best_result = self.gc.get_random_point(seed)
        last_best_result = current_best_result

        while temperature >= 0.1:
            last_best_result = current_best_result
            current_best_result = self.gc.disturb_point(current_best_result)

            (current_best_result, cost) = self.evaluate_point(
                current_best_result, last_best_result, temperature, penalty)

            self.cost_list.append(cost)
            self.results_list.append(current_best_result)

            temperature = self.reduce_temperature(temperature)
            self.it += 1

        print("(SA) Motivo de parada: Temperatura foi zerada!")
        finished_time = time()
        self.gc.plot_poits(self.cost_list, self.results_list)
        return self.results_list[self.cost_list.index(max(self.cost_list))], max(self.cost_list), finished_time

    def evaluate_point(self, best_result, last_best_result, T, penalty):
        current_cost = self.gc.func_g_x(best_result)
        last_cost = self.gc.func_g_x(last_best_result)
        probability = math.e**(last_cost - current_cost / T) - penalty

        if last_cost < current_cost:
            return (best_result, current_cost)
        elif random.random() <= probability:
            return (best_result, current_cost)
        else:
            return (last_best_result, last_cost)

    def reduce_temperature(self, T):
        # Fazer com função euleriana
        return T - (T/70)


def main():
    seed = 1588449669.2848306

    max_it = 500
    T = 500
    min_value = 0.9

    hc = HillClimbing()
    sa = SimulatedAnnealing()

    print("Semente: ", seed)
    start_hc_time = time()
    (hc_best_result, hc_cost, end_hc_time) = hc.run_hill_climbing(
        max_it, min_value, seed)
    hc_time = end_hc_time - start_hc_time

    start_sa_time = time()
    (sa_best_result, sa_cost, end_sa_time) = sa.run_simulated_annealing(T, seed)
    sa_time = end_sa_time - start_sa_time

    GenericClass.format_table(
        GenericClass, hc_best_result, hc_cost, sa_best_result, sa_cost)
    print("Tempo HC: ", hc_time)
    print("Tempo SA: ", sa_time)


if __name__ == "__main__":
    main()

# Fazer as respostas no readme
