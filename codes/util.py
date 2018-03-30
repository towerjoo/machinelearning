import os
import numpy as np

PATH = os.path.abspath(os.path.dirname(__file__))


def load_iris(cols=None, ratio=.8):
    """
    @params
    cols: cols to use, last col will be used as Y(target) 
    ratio: ratio between training set and test set

    @return
    a 5 elements tuple as (X_train, Y_train, X_test, Y_test, features)
    features is a list of feature names
    """
    path = os.path.join(PATH, "../dataset/iris.csv")
    targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    features = "sepal_length,sepal_width,petal_length,petal_width,class".split(",")
    if cols is not None:
        features = [features[i] for i in cols]
    # load dataset and also encode the 'class' feature as int to ease the processing
    iris = np.loadtxt(path, delimiter=",", usecols=cols, skiprows=1, converters={4: lambda x: targets.index(x)})
    # random choose the training set based on ratio
    rows, cols = iris.shape
    indexes = np.arange(rows)
    np.random.shuffle(indexes)
    train_index = indexes[:int(rows * ratio)]
    test_index = np.setdiff1d(np.arange(rows), train_index)
    train_set = iris[train_index]
    test_set = iris[test_index]
    return train_set[:, :-1], train_set[:, -1], test_set[:, :-1], test_set[:, -1], features

if __name__ == "__main__":
    print load_iris()

