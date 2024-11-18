import os
import numpy as np
import tarfile
import urllib.request
import random
import shutil
from pickle_tools import get_data_from_pickle, save_data_to_pickle
from typing import List, Tuple


class CIFAR10Pipeline():
    """
    Gère le téléchargement, le prétraitement et la séparation en ensembles d'entraînement et de test du jeu de données CIFAR-10


    Déroulement de la pipeline

            1) Le jeu de données CIFAR-10 sera tout d'abord téléchargé via l'URL "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
               en tant qu'archive sous le nom et le format "cifar-10-python.tar.gz".

            2) L'archive sera ensuite décompressée pour extraire les fichiers suivants : data_batch_1, data_batch_2,
               data_batch_3, data_batch_4, data_batch_5 et test_batch.
            
               Chacun de ces fichiers (format pickle), une fois chargée, représente un dictionnaire contenant les éléments suivants :

                   - 'data'   : Un tableau numpy de forme (10000, 3072) de type uint8. Chaque ligne de ce tableau représente une image couleur de taille 32x32.
		                        Les 1024 premières valeurs correspondent aux canaux rouges, les 1024 suivantes aux canaux verts, et les 1024 dernières aux canaux bleus.
		                        L'image est stockée en ordre de ligne, de sorte que les 32 premières valeurs de chaque ligne correspondent aux valeurs du canal rouge de la première ligne de l'image.

                   - 'labels' : Une liste de 10 000 entiers, chacun dans l'intervalle [0-9]. L'entier à l'index i indique l'étiquette (classe) de l'image correspondante dans le tableau 'data'.

            3) À partir de tous ces fichiers, on récupère ensuite une liste (de longueur 6 x 10 000 = 60 000).
               Chaque élément (x, y) de cette liste est un tuple où :

                   - x est un vecteur colonne (tableau 2D NumPy) de taille (3072, 1) représentant une image couleur

                   - y est un entier (int) compris entre 0 (inclus) et 9 (inclus) indiquant l'étiquette (classe) de l'image x
            
            4) On sépare la liste en deux : données d'entraînement et de test. On stocke respectivement ces deux nouvelles listes dans les fichiers training_data et test_data au format pickle.

            5) On supprime enfin les fichiers inutiles à partir de maintenant : cifar-10-python.tar.gz, data_batch_1, data_batch_2, data_batch_3, data_batch_4,
               data_batch_5 et test_batch.


    Attributs

        data_dir_path - str : Le chemin vers le répertoire où les données seront stockées
    """


    def __init__(self, data_dir_path: str):
        """
        Initialise la pipeline


        Entrées

            data_dir_path - str : Le chemin vers le répertoire où les données seront stockées
        """

        self.data_dir_path = data_dir_path


    def download_cifar10(self):
        """
        Étape 1 de la pipeline
        """

        print("Étape 1 de la pipeline en cours...")
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        tar_file_path = os.path.join(self.data_dir_path, "cifar-10-python.tar.gz")
        os.makedirs(self.data_dir_path, exist_ok=True)
        if not os.path.exists(tar_file_path):
            urllib.request.urlretrieve(url, tar_file_path)
        print("Étape 1 de la pipeline terminée\n")


    def extract_cifar10(self):
        """
        Étape 2 de la pipeline
        """

        print("Étape 2 de la pipeline en cours...")
        file_path = os.path.join(self.data_dir_path, "cifar-10-python.tar.gz")
        os.makedirs(self.data_dir_path, exist_ok=True)
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=self.data_dir_path)
        extracted_subdir_path = os.path.join(self.data_dir_path, "cifar-10-batches-py")
        for file in os.listdir(extracted_subdir_path):
            src = os.path.join(extracted_subdir_path, file)
            dest = os.path.join(self.data_dir_path, file)
            shutil.move(src, dest)
        os.rmdir(extracted_subdir_path)
        os.remove(os.path.join(self.data_dir_path, "readme.html"))
        os.remove(os.path.join(self.data_dir_path, "batches.meta"))
        print("Étape 2 de la pipeline terminée\n")


    def get_list_from_batch_files(self):
        """
        Étape 3 de la pipeline
        """

        print("Étape 3 de la pipeline en cours...")
        input_file_paths = [os.path.join(self.data_dir_path, "data_batch_1"), os.path.join(self.data_dir_path, "data_batch_2"), os.path.join(self.data_dir_path, "data_batch_3"), os.path.join(self.data_dir_path, "data_batch_4"), os.path.join(self.data_dir_path, "data_batch_5"), os.path.join(self.data_dir_path, "test_batch")]
        combined_data = []
        for file_path in input_file_paths:
            data = get_data_from_pickle(file_path)
            data = [(data[b'data'][i].reshape(3072, 1), data[b'labels'][i]) for i in range(10000)]
            combined_data.extend(data)
        print("Étape 3 de la pipeline terminée\n")
        return combined_data


    def split_data(self, data: List[Tuple[np.ndarray, int]], test_ratio: float):
        """
        Étape 4 de la pipeline


        Entrées

            data - list[tuple(np.ndarray, int)] : La liste obtenue à l'étape 3 de la pipeline

            test_ratio - float                  : La proportion des données à inclure dans l'ensemble de test (entre 0 et 1)

            train_file_path - str               : Le chemin pour sauvegarder l'ensemble d'entraînement au format pickle

            test_file_path - str                : Le chemin pour sauvegarder l'ensemble de test au format pickle
        """

        print("Étape 4 de la pipeline en cours...")
        train_file_path = os.path.join(self.data_dir_path, "training_data")
        test_file_path = os.path.join(self.data_dir_path, "test_data")
        np.random.shuffle(data)
        num_test = int(len(data) * test_ratio)
        test_data = data[:num_test]
        train_data = data[num_test:]
        save_data_to_pickle(train_data, train_file_path)
        save_data_to_pickle(test_data, test_file_path)
        print("Étape 4 de la pipeline terminée\n")


    def pipeline(self, test_ratio: float):
        self.download_cifar10()
        self.extract_cifar10()
        data = self.get_list_from_batch_files()
        self.split_data(data, test_ratio)
        print("Étape 5 de la pipeline en cours...")
        os.remove(os.path.join(self.data_dir_path, "cifar-10-python.tar.gz"))
        os.remove(os.path.join(self.data_dir_path, "data_batch_1"))
        os.remove(os.path.join(self.data_dir_path, "data_batch_2"))
        os.remove(os.path.join(self.data_dir_path, "data_batch_3"))
        os.remove(os.path.join(self.data_dir_path, "data_batch_4"))
        os.remove(os.path.join(self.data_dir_path, "data_batch_5"))
        os.remove(os.path.join(self.data_dir_path, "test_batch"))
        print("Étape 5 de la pipeline terminée")