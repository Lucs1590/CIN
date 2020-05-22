import pandas as pd


class AuxiliaryClass(object):

    def read_datas(self, path):
        return pd.read_csv(path)

    def define_dataframe_column(self, dataframe, columns):
        dataframe.columns = columns
        return dataframe

    def divide_dataset(self, dataset, train_percent, test_percent, validation_percent, seed):
        dataset = dataset.sample(frac=1, random_state=int(seed))
        train_dataset = dataset.iloc[
            0:
            round(len(dataset.index) * train_percent)
        ]
        test_dataset = dataset.iloc[
            len(train_dataset.index):
            len(train_dataset.index) + round(len(dataset.index) * test_percent)
        ]
        validation_dataset = dataset.iloc[
            len(test_dataset.index):
            len(test_dataset.index) + round(len(dataset.index) * validation_percent)
        ]
        return train_dataset, test_dataset, validation_dataset
