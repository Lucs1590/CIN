import numpy as np
from random import uniform


class Perceptron(object):

    def run_perceptron(self, train_ds, test_ds, validation_ds, max_it):
        train_ds = self.add_bias(train_ds)
        castes = self.define_castes(train_ds)
        neuron_castes = self.define_neuron(castes)

        (inputs, expected) = self.define_expecteds_and_inputs(train_ds)
        weights = self.define_weights(inputs, castes)

        predicted = self.activate_neurons(self.predict(train_ds, weights))
        errors = self.define_error(neuron_castes, expected, predicted)
        # entrar no loop para adaptar bias e erro

        it = 0

        while it <= max_it and 0 in errors:
            it += 1

    def add_bias(self, dataset):
        dataset['bias'] = 0
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
                list_of_weights.append(uniform(-1, 1))
                i += 1
            weights.append(list_of_weights)

        return np.array(weights)

    def predict(self, inputs, weights):
        weights_inputs = np.dot(inputs.values, weights)
        return weights_inputs

    def activate_neurons(self, predicted_values):
        row = 0
        while row < len(predicted_values):
            value = 0
            while value < len(predicted_values[row]):
                predicted_values[row][value] = 1 if predicted_values[row][value] >= 0 else 0
                value += 1
            row += 1
        return predicted_values

    def define_error(self, neuron_caste, expected, predicted):
        i = 0
        errors = []

        while i < len(expected):
            translated_expected = np.array(neuron_caste[expected.iloc[i]])
            error = 0
            i2 = 0

            while i2 < len(predicted[i]):
                error += 1 if predicted[i][i2] == translated_expected[i2] else 0
                i2 += 1
            errors.append(error)
            i += 1

        return errors
