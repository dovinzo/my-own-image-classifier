import numpy as np

class Network:
	"""
	Network - Représente un réseau de neurones de type perceptron multicouche (MLP)

	Informations:
		- Les vecteurs et les matrices sont représentés avec des tableaux bidimensionnels (2D) NumPy
		- Il n'y a pas de poids, de biais et de fonction d'activation pour la première couche

	Attributs:
		number_of_layers (int) - Le nombre de couches
		sizes (list) - La liste des tailles (int) de chaque couche
		biases (list) - La liste des biais (vecteur - tableau 2D NumPy) de chaque couche
		weights (list) - La liste des poids (matrice - tableau 2D NumPy) de chaque couche
		activation_function_names (list) - La liste des noms (str) de chaque fonction d'activation à utiliser pour chaque couche

	Méthodes:
		train - Entraîne le modèle sur les données d'entraînement

	Exemple:
		network = Network([2, 3, 10], ['sigmoid', 'sigmoid'])
	"""

	def __init__(self, sizes, activation_function_names):
		"""
		__init__ - Initialise une nouvelle instance de la classe Network

		Entrées:
			sizes (list) - La liste des tailles (int) de chaque couche
			activation_function_names (list) - La liste des noms (str) de chaque fonction d'activation à utiliser pour chaque couche
		"""

		self.number_of_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
		self.activation_function_names = activation_function_names

	def activation_function(self, z, activation_function_name):
		"""
		activation_function - Applique la fonction d'activation

		Entrées:
			z (vecteur - tableau 2D NumPy) - Le vecteur avant application de la fonction d'activation
			activation_function_name (str) - Le nom de la fonction d'activation à appliquer

		Sortie:
			activation (vecteur - tableau 2D NumPy) - Le vecteur après application de la fonction d'activation
		"""

		if activation_function_name == 'relu':
			return np.maximum(0, z)
		elif activation_function_name == 'sigmoid':
			return 1 / (1 + np.exp(-z))
		elif activation_function_name == 'tanh':
			return np.tanh(z)
		elif activation_function_name == 'softmax':
			exps = np.exp(z)
			return exps / np.sum(exps)
		else:
			raise ValueError(f"Fonction d'activation non prise en charge: {activation_function_name}")
	
	def feedforward(self, input):
		"""
		feedforward - Effectue la propagation avant (forward propagation) à travers toutes les couches du réseau

		Entrée:
			input (vecteur - tableau 2D NumPy) - Les données d'entrée du réseau

		Sortie:
			output (vecteur - tableau 2D NumPy) - Les sorties du réseau après la propagation avant
		"""

		activation = input

		for bias, weight, activation_function_name in zip(self.biases, self.weights, self.activation_function_names):
			activation = self.activation_function(np.dot(weight, activation) + bias, activation_function_name)

		output = activation

		return output
	
	def train(self, training_data, epochs, learning_rate, mini_batch_size):
		"""
		train - Entraîne le modèle sur les données d'entraînement

		Information: La méthode utilisée est "mini-batch stochastic gradient descent"

		Entrées:
			training_data (list) - La liste des données d'entraînement (x, y) (tuple), où :
							x (vecteur - tableau 2D NumPy) - Les données d'entrée
							y (vecteur - tableau 2D NumPy) - Les données de sortie souhaitées
			epochs (int) - Le nombre d'époques d'entraînement
			learning_rate (float) - Le taux d'apprentissage
			mini_batch_size (int) - La taille d'un mini-batch
		"""
