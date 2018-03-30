import numpy as np


"""The base class for ML methods
other concrete methods can extend this one and make the
corresponding implementation
"""

class ML(object):
    def __init__(self, name):
        self.name = name
        self.X_train, self.Y_train, self.X_test, self.Y_test, self.features = self.load_dataset()

    def load_dataset(self):
        raise NotImplementedError("need to implement in child class")

    def define(self):
        """define a model, e.g architecture, hyperparameters, etc.
        """
        pass

    def train(self, X, Y):
        """train a model
        """
        pass

    def predict(self, sample):
        """predict a data point, i.e sample
        """
        pass

    def evaluate(self, X, Y):
        """Evaluate the trained model, with unseen data points
        """
        pass

    def info(self):
        """print model's interesting information
        """
        pass

    def inspect(self):
        """inspect a model's useful information, e.g architecture, hyperparameters, etc.
        """
        print "Inspect {}:".format(self.name)
        
    def plot(self):
        """plot some interesting information about the model
        """
        pass
