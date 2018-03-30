import matplotlib.pyplot as plt
import numpy as np

from ml import ML
from util import load_iris


class kNN(ML):
    def __init__(self):
        super(kNN, self).__init__("kNN")

    def load_dataset(self):
        return load_iris(cols=[0, 1, 4])

    def inspect(self):
        super(kNN, self).inspect()
        targets = np.unique(self.Y_train)
        markers = ["ro", "bo", "yo"][:len(targets)]
        # plot based on different target
        data = []
        for i, t in enumerate(targets):
            index = np.argwhere(self.Y_train == t).ravel()
            data.extend([self.X_train[index][:, 0], self.X_train[index][:, 1], markers[i]])
        plt.plot(*data)
        plt.title("Iris dataset inspect")
        plt.xlabel(self.features[0])
        plt.ylabel(self.features[1])
        plt.show()

        
