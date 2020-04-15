import random
import time
import matplotlib.pyplot as plt
import math


def hill_climbing(max_it, min_value):
    it = 0
    cost = 0
    cost_list = []
    results_list = []
    current_best_result = get_random_point(time.time())
    last_best_result = current_best_result

    (current_best_result, cost) = evaluate_point(
        current_best_result, last_best_result)

    cost_list.append(cost)
    results_list.append(current_best_result)

    while it < max_it:
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
    # fazer o gausiano
    return point + 0.03


def evaluate_point(best_result, last_best_result):
    current_cost = func_g_x(best_result)
    last_cost = func_g_x(last_best_result)
    return (best_result, current_cost) if last_cost < current_cost else (last_best_result, last_cost)


def func_g_x(x):
    return (2**-2*(x-0.1/0.9)**2)*(math.sin(5*math.pi*x))**6


def plot_poits(data, data2):
    plt.plot(data2)
    plt.show()


def main():
    (best_result, cost) = hill_climbing(7000, 0.9)
    print('Melhor Resultado:', best_result)
    print('Custo: ', cost)


if __name__ == "__main__":
    main()
