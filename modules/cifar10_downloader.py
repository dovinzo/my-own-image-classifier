import os
import urllib.request

class CIFAR10Downloader:
    def __init__(self):
        path_to_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(path_to_data_dir, exist_ok=True)
        os.makedirs(os.path.join(path_to_data_dir, "raw"), exist_ok=True)
        self.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        self.path_to_tar_file = os.path.join(path_to_data_dir, "raw", "cifar-10-python.tar.gz")

    def download(self):
        if not os.path.exists(self.path_to_tar_file):
            urllib.request.urlretrieve(self.url, self.path_to_tar_file)
