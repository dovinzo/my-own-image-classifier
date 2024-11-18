from network import Network
import numpy as np

class Classifier(Network):
    """
    Classifier - Hérite de la classe Network et ajoute des fonctionnalités spécifiques à la classification.

    Méthodes supplémentaires :
        predict        : Prédit les classes pour les données d'entrée.
        evaluate       : Évalue la performance du classifieur sur des données de test.
    """

    def __init__(self, sizes, activation_function_names):
        """
        __init__ - Initialise une instance de la classe Classifier.
        Appelle le constructeur de la classe parent Network.
        """
        super().__init__(sizes, activation_function_names)
	
    def train(self, training_data, epochs, learning_rate, mini_batch_size):
        """
        Entraîne le classifieur sur des données où les sorties (y) sont des entiers.
        Transforme les étiquettes en vecteurs one-hot avant d'utiliser la méthode train de Network.

        Args:
            training_data (list): Liste des données d'entraînement (x, y) avec :
                - x : Vecteur d'entrée (tableau 2D NumPy).
                - y : Entier représentant la classe (sortie souhaitée).
            epochs (int): Nombre d'époques.
            learning_rate (float): Taux d'apprentissage.
            mini_batch_size (int): Taille des mini-batches.
        """
        # Déterminer le nombre total de classes à partir des y
        num_classes = max(y for _, y in training_data) + 1

        # Transformer y en vecteurs one-hot
        training_data_one_hot = []
        for x, y in training_data:
            y_one_hot = np.zeros((num_classes, 1))
            y_one_hot[y] = 1
            training_data_one_hot.append((x, y_one_hot))

        # Appeler la méthode train de la classe Network
        super().train(training_data_one_hot, epochs, learning_rate, mini_batch_size)

    def predict(self, input_vector):
        """
        predict - Prédit la classe pour une donnée d'entrée.

        Entrée :
            input_vector (vecteur / tableau 2D NumPy) : Les données d'entrée du réseau.

        Sortie :
            (int) : L'indice de la classe prédite (commence à 0).
        """
        output_vector = self.feedforward(input_vector)
        return int(np.argmax(output_vector))  # L'indice de la classe avec la probabilité maximale

    def evaluate(self, test_data):
        """
        evaluate - Évalue la performance du classifieur sur des données de test.

        Entrée :
            test_data (list) : Liste des tuples (x, y) où :
                - x (vecteur / tableau 2D NumPy) : Donnée d'entrée.
                - y (int)                       : Classe réelle (entier).

        Sortie :
            (float) : Précision du modèle sur les données de test.
        """
        test_results = [(self.predict(x), y) for x, y in test_data]
        accuracy = sum(int(pred == y) for pred, y in test_results) / len(test_data)
        return accuracy
