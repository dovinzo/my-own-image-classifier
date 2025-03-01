"""
download_cifar10.py

Ce script effectue le téléchargement du jeu de données CIFAR-10.
Le téléchargement se fait via l'URL "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz".
À l'issue du téléchargement, on récupère le fichier "cifar-10-python.tar.gz" dans le dossier "data/raw/".

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

from cifar10_downloader import CIFAR10Downloader

# ============================================================
#                       Début du script
# ============================================================

my_cifar10_downloader = CIFAR10Downloader()
my_cifar10_downloader.download()
