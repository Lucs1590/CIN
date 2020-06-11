import numpy as np
from random import uniform
import operator
from functools import reduce
from aux import AuxiliaryClass
# 1591900970.2312458


class Perceptron(object):
    def __init__(self):
        self.aux = AuxiliaryClass()

    def run_perceptron(self, train_ds, test_ds, validation_ds, max_it, learning_rate=0.01):
        print("Learning Rate: ", learning_rate)

        errors_list = []
        errors_avg_list = []

        train_ds = self.add_bias(train_ds)
        castes = self.define_castes(train_ds)
        neuron_castes = self.define_neuron(castes)

        (inputs, expected) = self.define_expecteds_and_inputs(train_ds)
        weights = self.define_weights(inputs, castes)

        error_sum = 1
        it = 0

        while it <= max_it and error_sum >= 0:
            i = 0
            error_sum = 0

            while i < len(train_ds):
                input_i = train_ds.values[i]

                input_weight = self.predict(input_i, weights)
                predicted = self.activate_neurons(input_weight)
                error = self.define_error(
                    neuron_castes, expected.values[i], predicted)

                if error > 0:
                    weights = self.update_weights(
                        weights, error, learning_rate, predicted)
                    train_ds = self.update_bias(error, learning_rate, train_ds)

                error_sum += error
                i += 1

            errors_list.append(error_sum)
            errors_avg_list.append(error_sum/len(train_ds))

            it += 1

        self.aux.show_results(errors_list, errors_avg_list)
        self.aux.plot_error(errors_list, errors_avg_list)

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
        expected = dataset.pop(column_name)
        return dataset, expected

    def define_weights(self, dataset, castes):
        weights = []
        for column in range(len(dataset.columns)):
            i = 0
            list_of_weights = []

            while i < len(castes):
                list_of_weights.append(uniform(0.001, 1))
                i += 1
            weights.append(list_of_weights)

        return np.array(weights)

    def predict(self, inputs, weights):
        weights_inputs = np.dot(inputs, weights)
        return weights_inputs

    def activate_neurons(self, predicted_values):
        value = 0
        while value < len(predicted_values):
            predicted_values[value] = 1 if predicted_values[value] >= 0 else 0
            value += 1
        return predicted_values

    def define_error(self, neuron_caste, expected, predicted):
        i = 0
        error = 0
        translated_expected = np.array(neuron_caste[expected])

        while i < len(predicted):
            error += 0 if predicted[i] == translated_expected[i] else 1
            i += 1
        return error

    def update_weights(self, weights, error, learning_rate, predicted):
        return weights + learning_rate * error * predicted

    def update_bias(self, error, learning_rate, dataset):
        dataset["bias"] = dataset["bias"].values + error + learning_rate
        return dataset
