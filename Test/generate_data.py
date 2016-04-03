__author__ = 'Kirill Rudakov'

from sklearn.datasets import make_classification

def generate_data(n_samples=10000, n_features = 20,weights = [0.3,1]):
    return make_classification(n_samples=n_samples, n_features=n_features, n_informative=3, n_classes=2,weights=weights)

