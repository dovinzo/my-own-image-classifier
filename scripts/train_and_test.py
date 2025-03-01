"""
train_and_test.py

Ce script effectue l'entraînement et le test du classifieur d'images.

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

from cifar10_loader import CIFAR10Loader
from image_classifier import ImageClassifier

# ============================================================
#                          Paramètres
# ============================================================

epochs = 10
learning_rate = 0.01
mini_batch_size = 64

# ============================================================
#                       Début du script
# ============================================================

my_cifar10_loader = CIFAR10Loader()
my_image_classifier = ImageClassifier()

training_data = my_cifar10_loader.load_training_data()
test_data = my_cifar10_loader.load_test_data()

X_train = training_data["X_train"]
Y_train = training_data["Y_train"]
X_test = test_data["X_test"]
Y_test = test_data["Y_test"]

my_image_classifier.train(X_train, Y_train, epochs, learning_rate, mini_batch_size)

accuracy = my_image_classifier.evaluate(X_test, Y_test)

print(f"accuracy : {accuracy}")
