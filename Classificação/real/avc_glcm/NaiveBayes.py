import numpy as np

## Data Partition
from avc_glcm_data_partition import Avc_glcm_data

dataSet = Avc_glcm_data()

X_train = dataSet.X_train_raw
X_test = dataSet.X_test_raw

y_train = dataSet.y_train_raw
y_test = dataSet.y_test_raw

## Naive Bayes
## Utilizando função gaussiana apos a analise do tipo de dados do data set(multiplas features, float).
def gaussian_nb():
    from sklearn.naive_bayes import GaussianNB

    GNB = GaussianNB()

    print("Started Training...")
    GNB.fit(X_train, y_train)

    print("\nStarted Testing...")
    # Predict training to check for overfitting
    y_train_prediction = GNB.predict(X_train)

    error = (len(y_train) - sum(y_train_prediction == y_train)) / len(y_train)

    print("\nTraining Data:")
    print(f"Training Error: {error}")
    print(f"Training Acurracy: {1 - error}")

    # Predict test
    y_test_prediction = GNB.predict(X_test)

    error = (len(y_test) - sum(y_test_prediction == y_test)) / len(y_test)

    print("\nTesting Data:")
    print(f"Testing Error: {error}")
    print(f"Testing Acurracy: {1 - error}") 

gaussian_nb()