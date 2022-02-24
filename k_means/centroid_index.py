"""
Criado por: Marcelo Jantsch Wille
Email: marcelojantschwille@gmail.com
Última modificação: 04/11/2021
Descrição: Validação da implementação do k-means pelo 'Centroid Index' (centróides órfãos).
"""

# Módulos de Python
from csv import reader
from copy import deepcopy
from math import inf

# Módulos do projeto
from k_means import K_means

# Arquivo com dados
DATA_PATH  = "../data/benchmark_instances.csv"
# Booleano que indica se arquivo possui cabeçalho
HAS_HEADER = False

# Arquivo com centróides 'ground truth'
CENTROIDS_PATH = "../data/benchmark_centroids_groundtruth.csv"

K = 15    # Valor de 'K' com o qual o algoritmo K-means será executado
I = 100   # Número de iterações para cálculo da menor distância intracluster de certo 'k'

def get_lowest_wss_centroids(k, original_data):
	"""
	Devido à inicialização aleatória dos centróides, roda
	o modelo um certo número de vezes para um determinado valor
	de 'k'. Retorna centróides do modelo com menor distância
	intracluster encontrada dentre todas essas iterações com esse 'k'.
	"""

	lowest_wss = inf
	lowest_centroids = {}

	# Gera modelo 'i' vezes para o valor de 'k' e calcula distâncias intracluster
	for _ in range(I):
		data = deepcopy(original_data)
		model = K_means(k, data, "euclidian")
		model.run()
		wss = model.get_wss()
		if wss < lowest_wss:
			lowest_wss = wss
			lowest_centroids = model.centroids

	return lowest_centroids    # Centróides do modelo com menor WSS


def centroid_distance(ci, cj):
	""" Calcula distância entre dois centróides, um da execução e um do ground truth """
	distance = 0
	for i, j in zip(ci, cj):
		distance += abs(i - j)

	return distance ** 2


if __name__ == '__main__':
	# Lê dados do arquivo
	data = []
	with open(DATA_PATH, 'r') as fp:
		csv_reader = reader(fp, delimiter=',')
		for line in csv_reader:
			data.append(line)

	# Eliminar headers, pois objeto K_means exige dados passados sem eles
	if HAS_HEADER:
		data = data[1:]

	# Executa K-means I vezes (devido a inicialização aleatória) e pega os centróides
	centroids = get_lowest_wss_centroids(K, data)

	# Imprime centróides
	print("Centróides:")
	for centroid in centroids.values():
		print(centroid['position'])
	print("-------------------------------------")

	# Faz cálculo do 'Centroid Index' ------------------------------------------

	# Lê do arquivo os centróides 'ground truth'
	ground_truths = []
	with open(CENTROIDS_PATH, 'r') as fp:
		csv_reader = reader(fp, delimiter=',')
		for line in csv_reader:
			ground_truths.append([float(n) for n in line])

	# Lista contendo quantos centróides do groud truth tem centróide considerados mais próximos
	# Um índice que permanece 0 é um centróide órfão
	orphans = [0] * len(ground_truths)

	# Calcula distâncias dos centróides mais próximos entre os centrídes
	# obtidos na execução e os centróides em groud truth
	for centroid in centroids.values():
		closest_centroid = {
			"index": -1,
			"distance": inf,
		}
		for i, ground_truth in enumerate(ground_truths):
			distance = centroid_distance(centroid['position'], ground_truth)
			if distance < closest_centroid['distance']:
				closest_centroid['index'] = i
				closest_centroid['distance'] = distance

		orphans[closest_centroid['index']] += 1

	# calcula dissimilaridade dos conjuntos de centróides contando o número de órfãos
	CI = orphans.count(0)
	print(f"Centroid Index = {CI}")
