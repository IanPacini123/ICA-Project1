import numpy as np
from typing import Callable

class MultiLayerPerceptron:

    def __init__(self, X_train: np.array, X_test: np.array, 
                 y_train: np.array, y_test: np.array,
                 activate_func: str='step', 
                 epochs: int=1000, learning_rate: float=0.1):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.activate_function = self.get_activate_func(activate_func)
        self.epochs = epochs
        self.learning_rate = learning_rate

    
    def get_activate_func(self, func: str) -> Callable:
        if func == 'sigmoid':
            return lambda matrix : 1 / (1 + np.exp(-matrix))
        elif func == 'softmax':
            exp_matrix = lambda matrix: np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
            return lambda matrix : exp_matrix(matrix) / np.sum(exp_matrix(matrix), axis=1, keepdims=True)   
        elif func == 'tanh':
            return lambda matrix : np.tanh(matrix)
        elif func == 'step':
            return lambda matrix : np.heaviside(matrix, 1)
        
    def train(self):
        weights = np.random.randn(self.y_train.shape[1], self.X_train.shape[1]) * 0.1
    
        print(weights)

        for epoch in range(self.epochs):
        
            z = self.X_train @ weights.T
            result = self.activate_function(z)
            error =  result - self.y_train
            grad = error / len(self.X_train)

            weights -= self.learning_rate * np.dot(grad.T, self.X_train)

            if epoch % 5 == 0:
                print(f"Epoch {epoch} / {self.epochs}")
            
        print("Training complete")
    
        print(weights)
        self.weights = weights

    def test(self):
        print(self.weights)
        y_test_prediction = self.predict(self.X_test, self.weights)
        
        error = (len(self.y_test) - sum(y_test_prediction == self.y_test)) / len(self.y_test)

        print(f"Error: {error[0]}")
        print(f"Acurracy: {1 - error[0]}") 
    
    def predict(self, X, weights: np.array) -> np.array:
        z = X @ weights.T

        result = self.activate_function(z)

        classe = np.argmax(result, axis=1)

        return np.expand_dims(classe, axis=1)
