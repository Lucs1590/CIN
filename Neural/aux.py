import pandas as pd 


class AuxiliaryClass(object):

    def read_datas(self, path):
        return pd.read_csv(path)