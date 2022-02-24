"""
Criado por: Marcelo Jantsch Wille
Email: marcelojantschwille@gmail.com
Última modificação: 03/11/2021
Descrição: Implementação do algoritmo k-means.
"""

from random import choice
from copy import deepcopy
from math import inf, sqrt
from statistics import mean, mode
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def euclidian_distance(pt1, pt2, data_types):
	"""
	Calcula a distância euclidiana entre 2 pontos de dados. Nesta implementação,
	esta função é sempre chamada com pt1 sendo a instância e pt2 sendo o centróide.
	"""
	distance = 0
	for i, [xi, xj] in enumerate(zip(pt1, pt2)):
		if data_types[i] == "numeric":
			distance += (xi - xj)**2
		else:
			distance += hamming_distance(xi, xj) ** 2
	return sqrt(distance)



def manhattan_distance(pt1, pt2, data_types):
	"""
	Calcula a distância manhattan entre 2 pontos de dados. Nesta implementação,
	esta função é sempre chamada com pt1 sendo a instância e pt2 sendo o centróide.
	"""
	distance = 0
	for i, [xi, xj] in enumerate(zip(pt1, pt2)):
		if data_types[i] == "numeric":
			distance += abs(xi - xj)
		else:
			distance += hamming_distance(xi, xj)
	return distance



def chebyshev_distance(pt1, pt2, data_types):
	"""
	Calcula a distância de chebyschev entre 2 pontos de dados. Nesta implementação,
	esta função é sempre chamada com pt1 sendo a instância e pt2 sendo o centróide.
	"""
	distances = []
	for i, [xi, xj] in enumerate(zip(pt1, pt2)):
		if data_types[i] == "numeric":
			distance = abs(xi - xj)
		else:
			distance = hamming_distance(xi, xj)
		distances.append(distance)
	return max(distances)



def hamming_distance(v1, v2):
	""" Calcula a distância de hamming entre 2 valores categóricos. """
	return 1 if v1 != v2 else 0



def is_numeric(str_value):
	""" Retorna True se valor é numérico (float ou int), False se for categórico. """
	try:
		a = float(str_value)
	except (TypeError, ValueError):
		try:
			b = int(str_value)
		except (TypeError, ValueError):
			return False
		else:
			return True
	else:
		return True



class K_means:
	""" Cria objetos capazes de rodar algoritmo k-means. """

	def __init__(self, k, data, distance = "euclidian"):

		self.k = k

		colors = [
			"darkgreen", "yellowgreen", "chartreuse",
			"yellow", "wheat", "silver", "goldenrod",
			"blue", "cyan", "teal", "maroon",
			"red", "tomato", "sienna", "pink",
			"fuchsia", "purple", "indigo", "black", "grey",
		]

		# Inicializa distância se está entre as possíveis,  senão levanta exceção.
		if distance == "euclidian":
			self.distance = euclidian_distance
		elif distance == "manhattan":
			self.distance = manhattan_distance
		elif distance == "chebyshev":
			self.distance = chebyshev_distance
		else:
			raise Exception("Distância especificada para algoritmo K-means é inválida.")

		# Toma os tipos (numérico ou categórico) de cada atributo
		self.data_types = []
		for value in data[0]:
			if is_numeric(value):
				self.data_types.append("numeric")
			else:
				self.data_types.append("categorical")

		# Pré-processamento dos dados: casting pra float de atributos numéricos
		processed_data = []
		for entry in data:
			processed_entry = []
			for value, data_type in zip(entry, self.data_types):
				if data_type == "numeric":
					processed_entry.append(float(value))
				else:
					processed_entry.append(value)
			processed_data.append(processed_entry)

		data = processed_data

		# Cada entrada de dados é uma lista. Coloca -1 na última posição de cada entrada
		# pois será usado como indicador do índice do centroide em 'self.centroids'
		self.data = []
		for entry in data:
			instance = deepcopy(entry)
			instance.append(-1)
			self.data.append(instance)

		# Inicializa centróides de forma aleatória a partir dos dados,
		# colocando os 'k' centróides em cima de 'k' pontos.
		self.centroids = {}
		for j in range(k):
			instance = choice(data)
			data.remove(instance)               # sem reposição
			self.centroids[j] = {}
			self.centroids[j]['position']  = instance
			self.centroids[j]['instances'] = []

			# Pode gerar cores para cada cluster somente se número de cores disponíveis menor que 'k'
			if k <= len(colors):
				color = choice(colors)
				colors.remove(color)
				self.centroids[j]['color'] = color
			else:
				self.centroids[j]['color'] = None


	def run(self, show_plots = False):
		""" Executa o algoritmo, implementando o loop principal do k-means. """

		instance_cluster_changed = True
		i = 1

		# Enquanto houver alteração nas associações de instâncias aos seus clusters
		while instance_cluster_changed:
			instance_cluster_changed = False

			# Apaga instâncias que pertenciam aos grupos dos centróides.
			# Vai incluir novamente as instãncias agora, no novo laço do algoritmo.
			self.clean_centroids_instances()

			# Para cada instância, encontra centróide mais próximo
			# instance[-1] é o centróide mais próximo da instância no último loop
			for instance in self.data:
				closest_centroid = self.find_closest_centroid(instance)
				self.centroids[closest_centroid]['instances'].append(instance)
				# Caso centróide mais próximo mudou, atualiza valor
				if closest_centroid != instance[-1]:
					instance[-1] = closest_centroid
					instance_cluster_changed = True

			# Pra cada centróide, corrige sua posição com base nas novas associações
			for j, centroid in self.centroids.items():
				self.update_centroid_position(j, centroid)

			# Plota gráfico com clusters formados nessa iteração (caso clusters tenham mudado)
			if show_plots and instance_cluster_changed:
				self.plot_clusters(i)

			i += 1


	def find_closest_centroid(self, instance):
		""" Retorna o índice do centróide mais próximo à instancia. """

		# Remove último valor (que representa o centróide) para fazer cálculo da distância
		instance_position = instance[:len(instance)-1]

		closest_centroid = {
			"index": -1,
			"distance": inf,
		}

		# Calcula o centróide mais próximo à instância passada como argumento
		for i, centroid in self.centroids.items():
			distance = self.distance(instance_position, centroid['position'], self.data_types)
			if distance < closest_centroid['distance']:
				closest_centroid['index'] = i
				closest_centroid['distance'] = distance

		return closest_centroid['index']


	def update_centroid_position(self, j, centroid):
		""" Calcula nova posição do centróide com base em suas instâncias. """

		new_position = []

		for i in range(len(self.data_types)):
			# Faz a média dos valores de cada atributo para ser essa a nova posição do centróide
			values = [attr[i] for attr in centroid['instances']]
			if self.data_types[i] == "numeric":
				attr_avg = mean(values)
			else:
				attr_avg = mode(values)
			new_position.append(attr_avg)

		self.centroids[j]['position'] = new_position


	def clean_centroids_instances(self):
		"""
		Apaga as instâncias que pertenciam aos grupos dos centróides
		para preencher novamente na próxima iteração do k-means.
		"""
		for j in self.centroids.keys():
			self.centroids[j]['instances'] = []


	def get_wss(self):
		""" Calcula e retorna valor da dissimilaridade intracluster do modelo. """

		centroid_distances = []

		# Calcula distâncias intracluster de cada grupo (ie. para cada centróide e suas instâncias)
		for centroid in self.centroids.values():
			distance = 0
			for instance in centroid['instances']:
				distance += self.distance(instance, centroid['position'], self.data_types) ** 2
			centroid_distances.append(distance)

		return sum(centroid_distances)   # Soma todas as distâncias intracluster


	def plot_clusters(self, i = 0):
		""" Plota gráfico com clusters formados. """

		# Não pode gerar plot se 'k' for maior que número de cores disponíveis
		if not self.centroids[0]['color']:
			raise Exception("Plot não foi possível. K é maior que número de cores.")

		plt.figure()

		# Gera pontos para cada cluster (assume dados 2D, pegando sempre os 2 primeiros valores)
		for centroid in self.centroids.values():
			cluster_x = []
			cluster_y = []
			for instance in centroid['instances']:
				cluster_x.append(instance[0])
				cluster_y.append(instance[1])

			# Plota pontos das instâncias
			plt.scatter(cluster_x, cluster_y, color = centroid['color'], marker = 'o', s =.2)

		plt.grid()

		# Se i != 0, significa que está plotando a cada iteração do k-means (enquanto cluster muda)
		if i != 0:
			plt.title(f"Clusters da Iteração {i} do K-means.")
		# Se i == 0, significa que somente está plotando o resultado final dos clusters encontrados
		else:
			plt.title("Clusters encontrados pelo K-means.")