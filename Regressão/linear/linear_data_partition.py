from pandas import read_csv
from numpy import array
from sklearn.model_selection import train_test_split

class Linear_data:
    def __init__(self) -> None:
        self.data_set = read_csv("DataSets/regressao/linear/data.csv", header=None)

        self.X = array(self.data_set.iloc[:, 0:-1])
        self.y = array(self.data_set.iloc[:, -1])

        ## Adding random_state helped stabilize the models,
        ## maybe something to do with how it stratifies?
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.X_train_raw = X_train_raw
        self.X_test_raw = X_test_raw

        self.y_train_raw = y_train_raw
        self.y_test_raw = y_test_raw