import numpy as np
from random import uniform
import operator
from functools import reduce
import math
from sklearn import preprocessing
import copy
from aux import AuxiliaryClass
from sklearn.metrics import confusion_matrix

# 1591900970.2312458


class Perceptron(object):
    def __init__(self):
        self.aux = AuxiliaryClass()

    def run_perceptron(self, train_ds, max_it, type_execution, learning_rate=0.0001):
        print("Learning Rate: ", learning_rate)

        lot_matix = []
        errors_avg_list = []
        real_error_list = []

        train_ds = self.add_bias(train_ds)
        castes = self.define_castes(train_ds)
        neuron_castes = self.define_neuron(castes)

        (inputs, expecteds) = self.define_expecteds_and_inputs(train_ds)
        weights = self.define_weights(inputs, castes)

        inputs = preprocessing.normalize(inputs)

        error_sum = 1
        it = 0

        while it < max_it and error_sum >= 0:
            i = 0
            error_sum = 0
            real_error_sum = 0
            correct_predictions = 0
            predicteds = []
            weights_list = []

            while i < len(inputs):
                input_i = inputs[i]

                input_weight = self.predict(input_i, weights)
                predicted, sigmoidal_values = self.activate_neurons(
                    input_weight)

                predicteds.append(copy.deepcopy(predicted))
                weights_list.append(weights)

                error, real_error = self.define_error(
                    neuron_castes, expecteds.values[i], predicted, sigmoidal_values)

                if type_execution == 'train':
                    if error > 0:
                        weights = self.update_weights(
                            weights, real_error, learning_rate, predicted)
                        inputs = self.update_bias(
                            real_error, learning_rate, inputs)
                    else:
                        correct_predictions += 1

                error_sum += error
                real_error_sum += real_error

                i += 1

            lot_matix.append(confusion_matrix(
                expecteds, self.translate_predicteds(predicteds, neuron_castes)))
            real_error_list.append(real_error_sum)
            errors_avg_list.append(error_sum/len(train_ds))

            it += 1

        min_error = real_error_list.index(min(real_error_list))
        best_weights = weights_list[min_error]
        best_conf_matrix = lot_matix[min_error]

        self.aux.show_results(
            real_error_list, errors_avg_list, correct_predictions, best_conf_matrix, best_weights)
        self.aux.plot_error(real_error_list, errors_avg_list)

    def add_bias(self, dataset):
        dataset['bias'] = 1
        return dataset

    def define_castes(self, dataset, column_name="class"):
        clusters = dataset[column_name].unique()
        return list(clusters)

    def define_neuron(self, castes):
        base = []
        castes_neuron = {}
        i = 0

        for caste in range(len(castes)):
            base.append([0] * len(castes))

        while i < len(castes):
            base[i][i] = 1
            castes_neuron[castes[i]] = base[i]
            i += 1

        return castes_neuron

    def define_expecteds_and_inputs(self, dataset, column_name="class"):
        final_dataset = copy.deepcopy(dataset)
        expected = final_dataset.pop(column_name)
        return final_dataset, expected

    def define_weights(self, dataset, castes):
        weights = []
        for column in range(len(dataset.columns)):
            i = 0
            list_of_weights = []

            while i < len(castes):
                list_of_weights.append(uniform(0.1, 1))
                i += 1
            weights.append(list_of_weights)

        return np.array(weights)

    def predict(self, inputs, weights):
        weights_inputs = np.dot(inputs, weights)
        return weights_inputs

    def activate_neurons(self, sigmoidal_values):
        value = 0
        sigmoidal_values = list(1 / (1 + math.e ** -sigmoidal_values))
        new_predicted_values = [0] * len(sigmoidal_values)
        new_predicted_values[sigmoidal_values.index(max(sigmoidal_values))] = 1
        return new_predicted_values, sigmoidal_values

    def define_error(self, neuron_caste, expected, predicted, sigmoidal_values):
        i = 0
        error = 0
        sum_real_error = 0
        translated_expected = np.array(neuron_caste[expected])

        while i < len(predicted):
            error += 0 if predicted[i] == translated_expected[i] else 1
            sum_real_error += abs(translated_expected[i] - sigmoidal_values[i])
            i += 1
        return error, float(sum_real_error)

    def update_weights(self, weights, error, learning_rate, predicted):
        i = 0
        while i < len(predicted):
            predicted[i] = error * learning_rate * predicted[i]
            i += 1
        return weights + predicted

    def update_bias(self, error, learning_rate, dataset):
        i = 0
        while i < len(dataset[:, -1]):
            dataset[:, -1][i] = dataset[:, -1][i] + (error * learning_rate)
            i += 1
        return dataset

    def translate_predicteds(self, predicted, castes):
        i = 0
        while i < len(predicted):
            for clss in castes:
                if predicted[i] == castes[clss]:
                    predicted[i] = clss
            i += 1

        return predicted

    def generate_matrix(self, expecteds, predicteds):
        return
