from pandas import read_csv
from numpy import array
from sklearn.model_selection import train_test_split

class Avc_glcm_data:
    def __init__(self) -> None:
        self.data_set = read_csv("DataSets/real/avc_glcm/data.csv")

        self.X = array(self.data_set.iloc[:, 1:-1])
        self.y = array(self.data_set.iloc[:, -1])

        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(self.X, self.y, stratify=self.y, test_size=0.2, random_state=42)

        self.X_train_raw = X_train_raw
        self.X_test_raw = X_test_raw

        self.y_train_raw = y_train_raw
        self.y_test_raw = y_test_raw

