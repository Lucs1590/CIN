import numpy as np


class Perceptron(object):

    def run_perceptron(self, train_ds, test_ds, validation_ds):
        train_ds = self.add_bias(train_ds)
        weights = self.define_weights(train_ds)
        castes = self.define_castes(train_ds)

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
