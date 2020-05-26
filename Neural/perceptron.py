import numpy as np


class Perceptron(object):

    def run_perceptron(self, train_ds, test_ds, validation_ds, max_it):
        train_ds = self.add_bias(train_ds)
        weights = self.define_weights(train_ds)
        castes = self.define_castes(train_ds)

        it = 0
        error = 0

        while it <= max_it and error > 0:
            it += 1

    def define_weights(self, dataset):
        weights = []
        for column in range(len(dataset.columns) - 1):
            weights.append(np.random.normal(0, 0.001, 1))
        return weights

    def add_bias(self, dataset):
        dataset['bias'] = 0
        return dataset

    def define_castes(self, dataset, column_name="class"):
        clusters = dataset[column_name].unique()
        return list(clusters)
