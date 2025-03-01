import numpy as np
import random
from typing import List

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu_derivative(z):
    return np.where(z <= 0, 0, 1)

def sigmoid_derivative(z):
    return np.exp(-z) / (1 + np.exp(-z))**2

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

class Network:
    """
    Représente un réseau de neurones de type perceptron multicouche (MLP).

    Les vecteurs colonnes sont représentés avec des tableaux 2D NumPy (np.ndarray).
    Les matrices sont représentés avec des tableaux 2D NumPy (np.ndarray).
    Il n'y a pas de poids, de biais et de fonction d'activation pour la première couche.

    Attributs
    ---------
    L - int
        Le nombre de couches.

    n - list[int]
        La liste des tailles de chaque couche : n = [n_1, n_2, ..., n_L].

    b - list[np.ndarray]
        La liste des biais de chaque couche : b = [b^2, b^3, ..., b^L].

    w - list[np.ndarray]
        La liste des poids de chaque couche : w = [w^2, w^3, ..., w^L].

    activation_function_names - list[str]
        La liste des noms de chaque fonction d'activation à utiliser pour chaque couche.
        Les noms utilisables sont : 'relu', 'sigmoid' et 'tanh'.

    Exemple
    -------
    >>> network = Network([2, 3, 10], ['sigmoid', 'sigmoid'])
    """

    def __init__(self, n: List[int], activation_function_names: List[str]):
        """
        Initialise une nouvelle instance de la classe Network.

        Entrées
        -------
        n - list[int]
            La liste des tailles de chaque couche : n = [n_1, n_2, ..., n_L].

        activation_function_names - list[str]
            La liste des noms de chaque fonction d'activation à utiliser pour chaque couche.
            Les noms utilisables sont : 'relu', 'sigmoid' et 'tanh'.
        """
        self.L = len(n)
        self.n = n
        self.b = [np.random.randn(y, 1) for y in n[1:]]
        self.w = [np.random.randn(y, x) for x, y in zip(n[:-1], n[1:])]
        self.activation_function_names = activation_function_names

    def __apply_activation_function(self, z: np.ndarray, activation_function_name: str) -> np.ndarray:
        """
        Applique la fonction d'activation.

        Entrées
        -------
        z - np.ndarray
            Le vecteur avant application de la fonction d'activation.

        activation_function_name - str
            Le nom de la fonction d'activation à appliquer.
            Le nom doit être : 'relu', 'sigmoid' ou 'tanh'.

        Sortie
        ------
        np.ndarray
            Le vecteur après application de la fonction d'activation.
        """
        if activation_function_name == 'relu':
            return relu(z)
        elif activation_function_name == 'sigmoid':
            return sigmoid(z)
        elif activation_function_name == 'tanh':
            return tanh(z)
        else:
            raise ValueError(f"Fonction d'activation non prise en charge : {activation_function_name}")

    def __apply_activation_function_derivative(self, z: np.ndarray, activation_function_name: str) -> np.ndarray:
        """
        Applique la dérivée de la fonction d'activation.

        Entrées
        -------
        z - np.ndarray
            Le vecteur avant application de la dérivée de la fonction d'activation.

        activation_function_name - str
            Le nom de la fonction d'activation dont on applique sa dérivée.
            Le nom doit être : 'relu', 'sigmoid' ou 'tanh'.

        Sortie
        ------
        np.ndarray
            Le vecteur après application de la dérivée de la fonction d'activation.
        """
        if activation_function_name == 'relu':
            return relu_derivative(z)
        elif activation_function_name == 'sigmoid':
            return sigmoid_derivative(z)
        elif activation_function_name == 'tanh':
            return tanh_derivative(z)
        else:
            raise ValueError(f"Fonction d'activation non prise en charge : {activation_function_name}")

    def feedforward(self, input_vector: np.ndarray):
        """
        Effectue la propagation avant à travers toutes les couches du réseau (forward propagation).

        Entrée
        ------
        input_vector : np.ndarray
            Le vecteur en entrée du réseau.

        Sortie
        ------
        output_vector : np.ndarray
            Le vecteur en sortie du réseau après la propagation avant.
        """
        a_l = input_vector
        for b_l, w_l, activation_function_name in zip(self.b, self.w, self.activation_function_names):
            a_l_moins_1 = a_l
            a_l = self.__apply_activation_function(np.dot(w_l, a_l_moins_1) + b_l, activation_function_name)
        output_vector = a_l
        return output_vector

    def train(self, training_data, epochs, learning_rate, mini_batch_size):
        """
        train - Entraîne le modèle sur les données d'entraînement

        Information :
        - La méthode utilisée est "mini-batch stochastic gradient descent".

        Entrées :
        training_data (list)  : La liste des données d'entraînement (x, y) (tuple) avec :
                                                x (vecteur / tableau 2D NumPy) : La donnée d'entrée
                                y (vecteur / tableau 2D NumPy) : La sortie souhaitée
        epochs (int)          : Le nombre d'époques d'entraînement
        learning_rate (float) : Le taux d'apprentissage
        mini_batch_size (int) : La taille d'un mini-batch
        """

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_parameters_mini_batch(mini_batch, learning_rate)
            print("Époque {0} complète".format(j))

    def calculate_cost_function_derivative(self, output_activation_vector, y):
        """
        calculate_cost_function_derivative - Calcule le vecteur dérivée partielle de la fonction de coût (pour une seule donnée d'entrée) par rapport à chaque coefficient du vecteur activation de la sortie

        Information :
        - La fonction de coût utilisée est l'erreur quadratique moyenne (MSE).

        Entrées :
        output_activation_vector (vecteur / tableau 2D NumPy) : Le vecteur activation de la sortie
        y (vecteur / tableau 2D NumPy)                        : La sortie souhaitée

        Sortie :
        (vecteur / tableau 2D NumPy) : Le résultat du calcul
        """
        return output_activation_vector - y

    def backpropagation(self, x, y):
        L = self.L

        bias_gradients = [np.zeros(b_l.shape) for b_l in self.b]
        weight_gradients = [np.zeros(w_l.shape) for w_l in self.w]

        a_l = x
        a = [x]
        z = []
        for b_l, w_l, activation_function_name in zip(self.b, self.w, self.activation_function_names):
            a_l_moins_1 = a_l
            z_l = np.dot(w_l, a_l_moins_1) + b_l
            z.append(z_l)
            a_l = self.__apply_activation_function(z_l, activation_function_name)
            a.append(a_l)

        error = self.calculate_cost_function_derivative(a[-1], y) * self.__apply_activation_function_derivative(z[-1], self.activation_function_names[-1])
        bias_gradients[-1] = error
        weight_gradients[-1] = np.dot(error, a[-2].transpose())

        for l in range(L-1, 1, -1):
            z_l = z[l-2]
            error = np.dot(self.w[l-1].transpose(), error) * self.__apply_activation_function_derivative(z_l, self.activation_function_names[l-2])
            bias_gradients[l-2] = error
            weight_gradients[l-2] = np.dot(error, a[l-2].transpose())

        return (bias_gradients, weight_gradients)

    def update_parameters_mini_batch(self, mini_batch, learning_rate):
        bias_gradients = [np.zeros(b_l.shape) for b_l in self.b]
        weight_gradients = [np.zeros(w_l.shape) for w_l in self.w]
        m = len(mini_batch)

        for x, y in mini_batch:
            bias_gradients_for_one_data, weight_gradients_for_one_data = self.backpropagation(x, y)
            bias_gradients = [bias_gradient + bias_gradient_for_one_data for bias_gradient, bias_gradient_for_one_data in zip(bias_gradients, bias_gradients_for_one_data)]
            weight_gradients = [weight_gradient + weight_gradient_for_one_data for weight_gradient, weight_gradient_for_one_data in zip(weight_gradients, weight_gradients_for_one_data)]

        self.b = [b_l - (learning_rate / m) * bias_gradient for b_l, bias_gradient in zip(self.b, bias_gradients)]
        self.w = [w_l - (learning_rate / m) * weight_gradient for w_l, weight_gradient in zip(self.w, weight_gradients)]
