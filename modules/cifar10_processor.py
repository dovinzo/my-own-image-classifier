import numpy as np
import os
from pickle_utils import load_data_from_pickle_file, save_data_to_pickle_file
import shutil
import tarfile
from typing import List, Tuple

class CIFAR10Processor:
    def __init__(self):
        path_to_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.path_to_tar_file = os.path.join(path_to_data_dir, "raw", "cifar-10-python.tar.gz")
        self.path_to_tmp_data_dir = os.path.join(path_to_data_dir, "tmp")
        os.makedirs(self.path_to_tmp_data_dir, exist_ok=True)
        os.makedirs(os.path.join(path_to_data_dir, "processed"), exist_ok=True)
        self.path_to_train_file = os.path.join(path_to_data_dir, "processed", "training_data")
        self.path_to_test_file = os.path.join(path_to_data_dir, "processed", "test_data")
        self.paths_to_training_batch_files = [os.path.join(self.path_to_tmp_data_dir, f"data_batch_{i}") for i in range(1, 6)]

    def extract(self):
        with tarfile.open(self.path_to_tar_file, "r:gz") as tar:
            tar.extractall(path=self.path_to_tmp_data_dir)
        path_to_extracted_subdir = os.path.join(self.path_to_tmp_data_dir, "cifar-10-batches-py")

        path_to_src_file = os.path.join(path_to_extracted_subdir, "data_batch_1")
        path_to_dest_file = os.path.join(self.path_to_tmp_data_dir, "data_batch_1")
        shutil.move(path_to_src_file, path_to_dest_file)

        path_to_src_file = os.path.join(path_to_extracted_subdir, "data_batch_2")
        path_to_dest_file = os.path.join(self.path_to_tmp_data_dir, "data_batch_2")
        shutil.move(path_to_src_file, path_to_dest_file)

        path_to_src_file = os.path.join(path_to_extracted_subdir, "data_batch_3")
        path_to_dest_file = os.path.join(self.path_to_tmp_data_dir, "data_batch_3")
        shutil.move(path_to_src_file, path_to_dest_file)

        path_to_src_file = os.path.join(path_to_extracted_subdir, "data_batch_4")
        path_to_dest_file = os.path.join(self.path_to_tmp_data_dir, "data_batch_4")
        shutil.move(path_to_src_file, path_to_dest_file)

        path_to_src_file = os.path.join(path_to_extracted_subdir, "data_batch_5")
        path_to_dest_file = os.path.join(self.path_to_tmp_data_dir, "data_batch_5")
        shutil.move(path_to_src_file, path_to_dest_file)

        path_to_src_file = os.path.join(path_to_extracted_subdir, "test_batch")
        path_to_dest_file = os.path.join(self.path_to_tmp_data_dir, "test_batch")
        shutil.move(path_to_src_file, path_to_dest_file)

        shutil.rmtree(path_to_extracted_subdir)

    def training_batch_files_to_training_data(self):
        my_dict = {"X_train": [], "Y_train": []}

        for path_to_training_batch_file in self.paths_to_training_batch_files:
            my_tmp_dict = load_data_from_pickle_file(path_to_training_batch_file)
            my_dict["X_train"].append(my_tmp_dict[b'data'])
            my_dict["Y_train"].append(np.array(my_tmp_dict[b'labels']).reshape(-1, 1))

        my_dict["X_train"] = np.concatenate(my_dict["X_train"], axis=0)
        my_dict["Y_train"] = np.concatenate(my_dict["Y_train"], axis=0)

        my_dict["X_train"] = my_dict["X_train"] / 255

        save_data_to_pickle_file(my_dict, self.path_to_train_file)

    def test_batch_file_to_test_data(self):
        my_dict = {"X_test": None, "Y_test": None}
        path_to_training_batch_file = os.path.join(self.path_to_tmp_data_dir, "test_batch")

        my_tmp_dict = load_data_from_pickle_file(path_to_training_batch_file)

        my_dict["X_test"] = my_tmp_dict[b'data']
        my_dict["Y_test"] = np.array(my_tmp_dict[b'labels']).reshape(-1, 1)

        my_dict["X_test"] = my_dict["X_test"] / 255

        save_data_to_pickle_file(my_dict, self.path_to_test_file)

    def remove_tmp_dir(self):
        shutil.rmtree(self.path_to_tmp_data_dir)
