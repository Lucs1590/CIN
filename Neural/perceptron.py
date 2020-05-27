import numpy as np


class Perceptron(object):

    def run_perceptron(self, train_ds, test_ds, validation_ds, max_it):
        train_ds = self.add_bias(train_ds)
        castes = self.define_castes(train_ds)
        weights = self.define_weights(train_ds, castes)

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

    def define_weights(self, dataset, castes):
        weights = []
        for column in range(len(dataset.columns) - 1):
            i = 0
            list_of_weights = []

            while i < len(castes):
                list_of_weights.append(float(np.random.normal(0, 0.001, 1)))
                i += 1
            weights.append(list_of_weights)

        return weights
