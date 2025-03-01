"""
process_cifar10.py

Ce script effectue le prétraitement du jeu de données CIFAR-10.
Voici les étapes principales :

1) Le fichier "cifar-10-python.tar.gz" est décompressé pour extraire les fichiers suivants dans le dossier "data/tmp/":
- "data_batch_1"
- "data_batch_2"
- "data_batch_3"
- "data_batch_4"
- "data_batch_5"
- "test_batch"

2) À partir des fichiers "batch" d'entraînement ("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"),
on récupère un dictionnaire Python avec les champs suivants :
- "X_train" : Un tableau 2D NumPy de taille (50 000, 3 072) correspondant aux 50 000 images d'entraînement, où chaque coefficient est un nombre entre 0 et 1.
- "Y_train" : Un tableau 2D NumPy de taille (50 000, 1) correspondant au label (chiffre entre 0 et 9) des 50 000 images d'entraînement.
Ce dictionnaire est ensuite sauvegardé en mémoire dans le fichier "data/processed/training_data".

3) À partir du fichier "batch" de test ("test_batch"), on récupère un dictionnaire Python avec les champs suivants :
- "X_test" : Un tableau 2D NumPy de taille (10 000, 3 072) correspondant aux 10 000 images de test.
- "Y_test" : Un tableau 2D NumPy de taille (10 000, 1) correspondant au label (chiffre entre 0 et 9) des 10 000 images de test.
Ce dictionnaire est ensuite sauvegardé en mémoire dans le fichier "data/processed/test_data".

4) Le dossier "data/tmp/" est supprimé.

Auteur : Kelvin Lefort
Date : 20 décembre 2024
"""

# ============================================================
#                   Importation des packages
# ============================================================

import os
import sys

if os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules") not in sys.path:
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules"))

from cifar10_processor import CIFAR10Processor

# ============================================================
#                       Début du script
# ============================================================

my_cifar10_processor = CIFAR10Processor()
my_cifar10_processor.extract()
my_cifar10_processor.training_batch_files_to_training_data()
my_cifar10_processor.test_batch_file_to_test_data()
my_cifar10_processor.remove_tmp_dir()
