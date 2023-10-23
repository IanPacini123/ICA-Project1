## Data Partition
from linear_data_partition import Linear_data

dataSet = Linear_data()

X_train = dataSet.X_train_raw
X_test = dataSet.X_test_raw

y_train = dataSet.y_train_raw
y_test = dataSet.y_test_raw

## SVR - RBF
def svr_rbf():
    from sklearn.metrics import mean_squared_error
    from sklearn.svm import SVR

    SVR = SVR(kernel='rbf')

    print("Started Training...")
    SVR.fit(X_train, y_train)

    print("\nStarted Testing...")

    y_test_prediction = SVR.predict(X_test)
    
    mse = mean_squared_error(y_test, y_test_prediction)

    print(mse)

## SVR - Polynomial
def svr_poly():
    from sklearn.metrics import mean_squared_error
    from sklearn.svm import SVR

    SVR = SVR(kernel='poly')

    print("Started Training...")
    SVR.fit(X_train, y_train)

    print("\nStarted Testing...")

    y_test_prediction = SVR.predict(X_test)
    
    mse = mean_squared_error(y_test, y_test_prediction)

    print(mse)

## SVR - Linear
def svr_linear():
    from sklearn.metrics import mean_squared_error
    from sklearn.svm import SVR

    SVR = SVR(kernel='linear')

    print("Started Training...")
    SVR.fit(X_train, y_train)

    print("\nStarted Testing...")

    y_test_prediction = SVR.predict(X_test)
    
    mse = mean_squared_error(y_test, y_test_prediction)

    print(mse)

svr_poly()