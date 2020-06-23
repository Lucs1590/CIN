import numpy as np
import matplotlib.pyplot as plt


class AuxiliaryClass(object):
    def __init__(self):
        ...

    def generate_population(self, numb_particles):
        ...


class PSOClass(object):
    def __init__(self):
        self.aux = AuxiliaryClass()

    def runPSO(self, particles, max_it, AC1, AC2, v_min, v_max):
        x = 0
        it = 0
        speed = self.generate_speed(v_min, v_max)

        while it < max_it:

            i = 0
            while i < len(particles):
                curr_particle = particles[i]
                best_optimal_indv = 0  # definir o melhor valor de todos que a particula já teve
                curr_optimal_indv = 0  # definir o melhor valor até hoje

                # colocar os valores comparados na função de aptidão
                if aux.fx(best_optimal_indv) > aux.fx(curr_optimal_indv):
                    curr_optimal_indv = best_optimal_indv

                neighbors = get_neighbors(curr_particle, particles)
                best_neighbor = get_best_neightbor(neighbors)

                for neighbor in neighbors:
                    optimal_of_neighbor = 0  # aplicar função de apitidão aqui onde ta 0
                    if aux.fx(optimal_of_neighbor) > aux.fx(best_neighbor):
                        best_neighbor = optimal_of_neighbor
                i += 1

                speed = self.update_speed()

            it += 1

    def generate_speed(self, v_min, v_max):
        ...

    def get_neighbors(self, curr_particle, flock):
        ...

    def get_best_neightbor(self, neighbors):
        ...

    def update_speed(self):
        ...


class ACOClass(object):
    def __init__(self):
        self.aux = AuxiliaryClass()

    def run_ACO(self):
        ...


def main():
    ...


if __name__ == "__main__":
    main()
