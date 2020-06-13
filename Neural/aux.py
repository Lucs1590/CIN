import pandas as pd
import matplotlib.pyplot as plt


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
            len(test_dataset.index) +
                round(len(dataset.index) * validation_percent)
        ]
        return train_dataset, test_dataset, validation_dataset

    def plot_error(self, errors, error_avg):
        df = pd.DataFrame({"errors": errors, "error_avg": error_avg})
        plt.subplot(211)
        plt.plot("errors", data=df, color="red")
        plt.title("Erros & Error Avg.")
        plt.ylabel("Error")

        plt.subplot(212)
        plt.plot("error_avg", data=df, color="green")
        plt.xlabel("Interactions")
        plt.ylabel("Error Avg.")
        plt.show()

    def show_results(self, errors_list, errors_avg_list, correct_predictions, best_conf_matrix, best_weights):
        print("Min. Error: ", min(errors_list))
        print("Min. Avg. Error: ", min(errors_avg_list))
        print("Last Correct Predictions: ", correct_predictions)
        print("Best Conf. Matrix: ", best_conf_matrix)
        print("Best Weight: ", best_weights)
