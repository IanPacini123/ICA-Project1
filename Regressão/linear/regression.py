## Data Partition
from linear_data_partition import Linear_data

dataSet = Linear_data()

X_train = dataSet.X_train_raw
X_test = dataSet.X_test_raw

y_train = dataSet.y_train_raw
y_test = dataSet.y_test_raw

def linear_regression():
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression

    LNR = LinearRegression()

    LNR.fit(X_train, y_train)

    y_test_prediction = LNR.predict(X_test)

    mse = mean_squared_error(y_test, y_test_prediction)

    print(mse)

linear_regression()