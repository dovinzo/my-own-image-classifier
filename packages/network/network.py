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
		activation_function - Applique la fonction d'activation
		activation_function_derivative - Applique la fonction d'activation dérivée
		feedforward - Effectue la propagation avant (forward propagation) à travers toutes les couches du réseau
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

		Information:
			- Voici la liste des noms possibles pour la fonction d'activation :
				'relu'
				'sigmoid'
				'tanh'

		Entrées:
			z (vecteur - tableau 2D NumPy) - Le vecteur avant application de la fonction d'activation
			activation_function_name (str) - Le nom de la fonction d'activation à appliquer

		Sortie:
			(vecteur - tableau 2D NumPy) - Le vecteur après application de la fonction d'activation
		"""

		if activation_function_name == 'relu':
			return np.maximum(0, z)
		elif activation_function_name == 'sigmoid':
			return 1 / (1 + np.exp(-z))
		elif activation_function_name == 'tanh':
			return np.tanh(z)
		else:
			raise ValueError(f"Fonction d'activation non prise en charge: {activation_function_name}")

	def activation_function_derivative(self, z, activation_function_name):
		"""
		activation_function_derivative - Applique la fonction d'activation dérivée

		Information:
			- Voici la liste des noms possibles pour la fonction d'activation :
				'relu'
				'sigmoid'
				'tanh'

		Entrées:
			z (vecteur - tableau 2D NumPy) - Le vecteur avant application de la fonction d'activation dérivée
			activation_function_name (str) - Le nom de la fonction d'activation dont on applique sa dérivée

		Sortie:
			(vecteur - tableau 2D NumPy) - Le vecteur après application de la fonction d'activation dérivée
		"""

		if activation_function_name == 'relu':
			return np.where(z <= 0, 0, 1)
		elif activation_function_name == 'sigmoid':
			return np.exp(-z) / (1 + np.exp(-z))**2
		elif activation_function_name == 'tanh':
			return 1 - np.tanh(z)**2
		else:
			raise ValueError(f"Fonction d'activation non prise en charge: {activation_function_name}")

	def feedforward(self, input_vector):
		"""
		feedforward - Effectue la propagation avant (forward propagation) à travers toutes les couches du réseau

		Entrée:
			input_vector (vecteur - tableau 2D NumPy) - Les données d'entrée du réseau

		Sortie:
			output_vector (vecteur - tableau 2D NumPy) - Les sorties du réseau après la propagation avant
		"""

		activation = input_vector

		for bias, weight, activation_function_name in zip(self.biases, self.weights, self.activation_function_names):
			activation = self.activation_function(np.dot(weight, activation) + bias, activation_function_name)

		output_vector = activation

		return output_vector
	
	def train(self, training_data, epochs, learning_rate, mini_batch_size):
		"""
		train - Entraîne le modèle sur les données d'entraînement

		Information: La méthode utilisée est "mini-batch stochastic gradient descent"

		Entrées:
			training_data (list) - La liste des données d'entraînement (x, y) (tuple), où :
							x (vecteur - tableau 2D NumPy) - La donnée d'entrée
							y (vecteur - tableau 2D NumPy) - La donnée de sortie souhaitée
			epochs (int) - Le nombre d'époques d'entraînement
			learning_rate (float) - Le taux d'apprentissage
			mini_batch_size (int) - La taille d'un mini-batch
		"""

	def cost_function_derivative(self, output_activation_vector, y):
		"""
		cost_function_derivative - Calcule le vecteur dérivée partielle de la fonction de coût (pour une seule donnée d'entrée) par rapport à chaque coefficient du vecteur activation de la sortie

		Information:
			- La fonction de coût utilisée est l'erreur quadratique moyenne (MSE)

		Entrées:
			output_activation_vector (vecteur - tableau 2D NumPy) - Le vecteur activation de la sortie
			y (vecteur - tableau 2D NumPy) - Le vecteur de la donnée de sortie souhaitée

		Sortie:
			(vecteur - tableau 2D NumPy) - Le résultat du calcul
		"""

		return output_activation_vector - y

	def backpropagation(self, x, y):
		bias_gradients = [np.zeros(bias.shape) for bias in self.biases]
		weight_gradients = [np.zeros(weight.shape) for weight in self.weights]

		activation = x
		activations = [x]
		weighted_inputs = []
		for bias, weight, activation_function_name in zip(self.biases, self.weights, self.activation_function_names):
			weighted_input = np.dot(weight, activation) + bias
			weighted_inputs.append(weighted_input)
			activation = self.activation_function(weighted_input, activation_function_name)
			activations.append(activation)

		error = self.cost_function_derivative(activations[-1], y) * self.activation_function_derivative(weighted_inputs[-1], self.activation_function_names[-1])
		bias_gradients[-1] = error
		weight_gradients[-1] = np.dot(error, activations[-2].transpose())

		for l in range(L-1, 1, -1):
			weighted_input = weighted_inputs[l-1]
			error = np.dot(self.weights[l].tanspose(), error) * self.activation_function_derivative(weighted_input, self.activation_function_names[l-1])
			bias_gradients[l-1] = error
			weight_gradients[l-1] = np.dot(error, activations[l-2].transpose())

		return (bias_gradients, weight_gradients)
