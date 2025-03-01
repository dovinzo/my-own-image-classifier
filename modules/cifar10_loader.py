import os
from pickle_utils import load_data_from_pickle_file

class CIFAR10Loader:
    def __init__(self):
        self.path_to_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    def load_training_data(self):
        return load_data_from_pickle_file(os.path.join(self.path_to_data_dir, "processed", "training_data"))

    def load_test_data(self):
        return load_data_from_pickle_file(os.path.join(self.path_to_data_dir, "processed", "test_data"))
