import numpy as np


class Perceptron(object):

    def run_perceptron(self, train_ds, test_ds, validation_ds, max_it):
        train_ds = self.add_bias(train_ds)
        castes = self.define_castes(train_ds)
        (inputs, expected) = self.define_expecteds_and_inputs(train_ds)
        weights = self.define_weights(inputs, castes)

        abc = self.predict(train_ds, weights)

        it = 0

        error = 0

        while it <= max_it and error >= 0:
            it += 1

    def add_bias(self, dataset):
        dataset['bias'] = 0
        return dataset

    def define_castes(self, dataset, column_name="class"):
        clusters = dataset[column_name].unique()
        return list(clusters)

    def define_expecteds_and_inputs(self, dataset, column_name="class"):
        expected = dataset.pop(column_name)
        return dataset, expected

    def define_weights(self, dataset, castes):
        weights = []
        for column in range(len(dataset.columns)):
            i = 0
            list_of_weights = []

            while i < len(castes):
                list_of_weights.append(float(np.random.normal(0, 0.001, 1)))
                i += 1
            weights.append(list_of_weights)

        return np.array(weights)

    def predict(self, inputs, weights):
        weights_inputs = np.dot(inputs.values, weights)
        return weights_inputs.sum(axis=1)
