import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Algorithms import MultiLayerPerceptron

data_set = pd.read_csv("C:/Users/ianpa/OneDrive/Área de Trabalho/VSCode/ICA-Project1/DataSets/real/avc_glcm/data.csv")
print(data_set)

X = np.array(data_set.iloc[:, 1:-1])
y = np.array(data_set.iloc[:, -1])

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, stratify=y, test_size=0.2)

X_train = np.column_stack((X_train_raw, np.ones(X_train_raw.shape[0])))
X_test = np.column_stack((X_test_raw, np.ones(X_test_raw.shape[0])))

y_train = y_train_raw.reshape(y_train_raw.shape[0], -1)
y_test = y_test_raw.reshape(y_test_raw.shape[0], -1)

MLP = MultiLayerPerceptron(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, activate_func='step', epochs=5000)

MLP.train()
MLP.test()
