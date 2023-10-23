import numpy as np

## Data Partition
from car_data_partition import Car_data

dataSet = Car_data()

X_train = dataSet.X_train_raw
X_test = dataSet.X_test_raw

y_train = dataSet.y_train_raw
y_test = dataSet.y_test_raw

## MLP
def mlp():
    from sklearn.neural_network import MLPClassifier

    X_train_w_bias = np.column_stack((X_train, np.ones(X_train.shape[0])))
    X_test_w_bias = np.column_stack((X_test, np.ones(X_test.shape[0])))

    MLP = MLPClassifier(hidden_layer_sizes=int(dataSet.X.shape[0]/2), activation='relu', solver='adam', max_iter=200, learning_rate_init=0.1, random_state=42)

    print("Started Training...")
    MLP.fit(X_train_w_bias, y_train)
    
    print("\nStarted Testing...")
    # Predict training to check for overfitting
    y_train_prediction = MLP.predict(X_train_w_bias)

    error = (len(y_train) - sum(y_train_prediction == y_train)) / len(y_train)

    print("\nTraining Data:")
    print(f"Training Error: {error}")
    print(f"Training Acurracy: {1 - error}")

    # Predict test
    y_test_prediction = MLP.predict(X_test_w_bias)

    error = (len(y_test) - sum(y_test_prediction == y_test)) / len(y_test)

    print("\nTesting Data:")
    print(f"Testing Error: {error}")
    print(f"Testing Acurracy: {1 - error}")

mlp()