import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from Algorithms import MultiLayerPerceptron

data_set = pd.read_csv("DataSets/real/avc_glcm/data.csv")
print(data_set)

X = np.array(data_set.iloc[:, 1:-1])
y = np.array(data_set.iloc[:, -1])

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, stratify=y, test_size=0.2)

X_train = np.column_stack((X_train_raw, np.ones(X_train_raw.shape[0])))
X_test = np.column_stack((X_test_raw, np.ones(X_test_raw.shape[0])))

y_train = y_train_raw
y_test = y_test_raw

MLP = MLPClassifier(hidden_layer_sizes=120, activation='relu', solver='adam', max_iter=3500, learning_rate_init=0.1, random_state=42)

MLP.fit(X_train, y_train)

y_test_prediction = MLP.predict(X_test)

error = (len(y_test) - sum(y_test_prediction == y_test)) / len(y_test)

print(f"Error: {error}")
print(f"Acurracy: {1 - error}") 
print(int(data_set.shape[0]/4))

#MLP = MultiLayerPerceptron(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, activate_func='step', epochs=10000)
#
#MLP.train()
#MLP.test()
