from pickle_tools import get_data_from_pickle, save_data_to_pickle
import os

class CIFAR10Loader():
    def __init__(self, data_dir_path: str):
        self.data_dir_path = data_dir_path
	
    def get_training_data(self):
        return get_data_from_pickle(os.path.join(self.data_dir_path, "training_data"))
    
    def get_test_data(self):
        return get_data_from_pickle(os.path.join(self.data_dir_path, "test_data"))