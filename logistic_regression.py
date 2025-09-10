# Import libraries
import numpy as np 

class Logistic_Regression_GD:

    def __init__(self, learning_rate=0.01, num_iter=100):
        """
        Initializes logisitic regression with gradient descent
        model.
        Parameters:
            - Learning Rate - Rate at which weights will be updated. Default value is 0.01 (float)
            - num_iter - The number of iterations to train. Default value is 100 (int)
        """
        self._learning_rate = learning_rate 
        self._num_iter = num_iter
        self._weights = None 
    
    def __sigmoid(self, value):
        """
        Computes the sigmoid value of value
        Parameters:
          - value: The number passed into sigmoid function.
        Returns:
          - result: The result of passing value into sigmoid function.
        """

        result = 1 / (1 + np.exp((-1 * value)))
        return result 

    
    def fit(self, X, Y):
        """
        Trains using logistic regression algorithm 
        Parameters:
            - X: Input data (NumPy Matrix)
            - Y: Output data/expected values (NumPy array)
        """

        # Initialize a random set of weights 
        self._weights = np.random.randn(1, X.shape[1]).reshape(-1)

        # Iterate the number of loops specified
        for i in range(0, self._num_iter):
            for j in range(0, len(X)):
                # Get the corresponding Y value (0 or 1)
                output = Y[j]

                # Compute the weighted sum
                weighted_sum = np.dot(self._weights, X[j])

                # Get the sigmoid value
                activation = self.__sigmoid(weighted_sum)

                # Compute the gradient 
                gradient = X[j] * (activation - output)

                # Update the weights 
                self._weights = self._weights - (self._learning_rate * gradient)
                
        # Return the trained weights 
        return self._weights 

    
    def predict(self, X):
        """
        Predicts the sentiment for tweet using the learned rates 
        during training.
        Parameters:
          - X: Input data
        Returns:
          - Sentiment: 0 or 1 depending on sentiment of tweet
        """

        sentiment = 0
        # Compute the activation value
        value = self.__sigmoid(self._weights.dot(X))
        # Apply threshold to get sentiment
        if value >= 0.5:
          sentiment = 1
        # Return the sentiment
        return sentiment







