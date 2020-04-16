import random
from time import time
# 1586998890.1230004
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd


def hill_climbing(max_it, min_value):
    it = 0
    it_repeat = 0
    cost = 0
    cost_list = []
    results_list = []
    current_best_result = get_random_point(time())  # time())
    last_best_result = current_best_result

    (current_best_result, cost) = evaluate_point(
        current_best_result, last_best_result)

    cost_list.append(cost)
    results_list.append(current_best_result)

    while it < max_it:
        """coment """
        it_repeat += 1 if current_best_result == last_best_result else 0
        if (it_repeat == (max_it/4)):
            print('Sem melhorias!')
            break
        last_best_result = current_best_result
        current_best_result = disturb_point(current_best_result)
        (current_best_result, cost) = evaluate_point(
            current_best_result, last_best_result)

        cost_list.append(cost)
        results_list.append(current_best_result)

        it += 1
    plot_poits(cost_list, results_list)
    return current_best_result, cost


def get_random_point(seed):
    print(seed)
    random.seed(seed)
    return round(random.random(), 2)


def disturb_point(point):
    return float(point + abs(np.random.normal(0, 0.001, 1)))


def evaluate_point(best_result, last_best_result):
    current_cost = func_g_x(best_result)
    last_cost = func_g_x(last_best_result)
    return (best_result, current_cost) if last_cost < current_cost else (last_best_result, last_cost)


def func_g_x(x):
    return (2**-2*(x-0.1/0.9)**2)*(math.sin(5*math.pi*x))**6


def plot_poits(data_cost, data_result):
    df = pd.DataFrame({'custo': data_cost, 'resultado': data_result})
    plt.subplot(211)
    plt.plot('custo', data=df, color='red')
    plt.title('Custo e Resultado')
    plt.ylabel('Custo')

    plt.subplot(212)
    plt.plot('resultado', data=df, color='green')
    plt.xlabel('Interações')
    plt.ylabel('Resultados')
    # plt.legend()
    plt.show()


def main():
    (best_result, cost) = hill_climbing(500, 0.9)
    print('Melhor Resultado:', best_result)
    print('Custo: ', cost)


if __name__ == "__main__":
    main()

""" if current_best_result > 1:
            current_best_result = results_list[-2]
            cost = cost_list[-2]
            cost_list.pop()
            results_list.pop()
            break """
