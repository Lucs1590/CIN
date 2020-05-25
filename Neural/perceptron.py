import numpy as np


class Perceptron(object):

    def run_perceptron(self, train_ds, test_ds, validation_ds):
        weights = self.define_weights(train_ds)

    def activation_func(self, sample, weights):
        activation = weights[0]
        for i in range(len(sample)-1):
            activation += weights[i + 1] * sample[i]
        return 1.0 if activation >= 0.0 else 0.0

    def define_weights(self, dataset):
        weights = []
        for column in range(len(dataset.columns) - 1):
            weights.append(np.random.normal(0, 0.001, 1))
        return weights
