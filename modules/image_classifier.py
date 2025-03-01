from network import Network
import numpy as np

class ImageClassifier(Network):
    """
    Hérite de la classe Network et ajoute des fonctionnalités spécifiques à la classification d'images.

    Attributs supplémentaires
    -------------------------
    label_names - list[str]
        La liste ["avion", "automobile", "oiseau", "chat", "cerf", "chien", "grenouille", "cheval", "navire", "camion"] correspondant aux classes/labels.

    Méthodes supplémentaires
    ------------------------
    predict
        Prédit la classe d'une image.
    evaluate
        Évalue la performance du classifieur sur les données de test.
    """

    def __init__(self):
        super().__init__(n=[3072, 100, 10], activation_function_names=['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'])

    def train(self, X_train, Y_train, epochs, learning_rate, mini_batch_size):
        """
        Entraîne le classifieur sur les données d'entraînement.

        Entrées
        -------
        X_train - np.ndarray
            Le tableau 2D NumPy de taille (50 000, 3 072) correspondant aux images d'entraînement.
            Chaque coefficient de ce tableau est un nombre entre 0 et 1.

        Y_train - np.ndarray
            Le tableau 2D NumPy de taille (50 000, 1) correspondant au label des images d'entraînement.
            Chaque coefficient de ce tableau est un entier entre 0 et 9.

        epochs - int
            Le nombre d'époques

        learning_rate - float
            Le taux d'apprentissage

        mini_batch_size - int
            La taille des mini-batches
        """
        training_data = []
        for i in range(50000):
            vecteur = np.zeros((10, 1), dtype=float)
            vecteur[Y_train[i, 0]] = 1.0
            training_data.append((X_train[i, :].reshape(-1, 1), vecteur))
        super().train(training_data, epochs, learning_rate, mini_batch_size)

    def predict(self, X):
        """
        Prédit la classe pour des images.

        Entrée
        ------
        X - np.ndarray
            Un tableau 2D NumPy avec 3 072 colonnes correspondant aux images dont on veut prédire leur classe.

        Sortie
        ------
        Y_hat - np.ndarray
            Un tableau 2D NumPy avec 1 colonne correspondant aux classes prédites par le modèle des images.
            Chaque coefficient de ce tableau est un entier entre 0 et 9.
        """
        Y_hat = np.array([np.argmax(super().feedforward(x.reshape(-1, 1)).ravel()) for x in X]).reshape(-1, 1)
        return Y_hat

    def evaluate(self, X_test, Y_test):
        """
        Évalue la performance du classifieur sur des données de test.

        Entrées
        -------
        X_test - np.ndarray
            Le tableau 2D NumPy de taille (10 000, 3 072) correspondant aux images de test.
            Chaque coefficient de ce tableau est un nombre entre 0 et 1.

        Y_test - np.ndarray
            Le tableau 2D NumPy de taille (10 000, 1) correspondant au label des images de test.
            Chaque coefficient de ce tableau est un entier entre 0 et 9.
     
        Sortie
        ------
        float
            Précision du modèle sur les données de test (nombre entre 0 et 1).
        """
        Y_test_hat = self.predict(X_test)
        accuracy = sum(int(y_test_hat == y_test) for y_test_hat, y_test in zip(Y_test_hat.ravel(), Y_test.ravel())) / 10000
        return accuracy
