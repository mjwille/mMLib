"""
Criado por: Marcelo Jantsch Wille
Email: marcelojantschwille@gmail.com
Última modificação: 04/11/2021
Descrição: Faz análise exploratória do dataset bank_t2,
plotando clusters gerados na execução do k-means.
"""

# Módulos de Python
from csv import reader
from copy import deepcopy
from itertools import combinations
import matplotlib.pyplot as plt

# Módulos do projeto
from k_means import K_means

# Arquivo com dados
DATA_PATH  = "../data/bank_t2.csv"

# TODO substituir pelo 'k' ideal do 'find_best_k.py'
K = 3     # Valor de 'K' com o qual o algoritmo K-means será executado
I = 3     # Número de iterações para cálculo da menor distância intracluster de certo 'k'

def get_lowest_wss_model(k, original_data):
	"""
	Devido à inicialização aleatória dos centróides, roda
	o modelo um certo número de vezes para um determinado valor
	de 'k'. Retorna o modelo com menor distância intracluster
	encontrada dentre todas essas iterações com esse 'k'.
	"""

	data = deepcopy(original_data)
	model = K_means(k, data)
	model.run()

	lowest_wss_model = model

	# Gera modelo 'i' vezes para o valor de 'k' e calcula distâncias intracluster
	for _ in range(I):
		data = deepcopy(original_data)
		model = K_means(k, data)
		model.run()
		if model.get_wss() < lowest_wss_model.get_wss():
			lowest_wss_model = model

	return lowest_wss_model   # menor distância intracluster encontrada para esse valor de 'k'


if __name__ == '__main__':
	# Lê dados do arquivo
	data = []
	with open(DATA_PATH, 'r') as fp:
		csv_reader = reader(fp, delimiter=',')
		for line in csv_reader:
			data.append(line)

	# Pega cabeçalhos (nomes dos atributos) para depois plotar os gráficos
	attrs = []
	for i, attr in enumerate(data[0]):
		attrs.append((i, attr))

	# Executa K-means
	model = get_lowest_wss_model(K, data[1:])

	# Plota clusters em gráficos 2D, casando atributo com atributo
	for attr_x, attr_y in combinations(attrs, 2):
		plt.figure()
		xi = attr_x[0]
		x_label = attr_x[1]
		yi = attr_y[0]
		y_label = attr_y[1]
		# Pega clusters (com as instâncias) dos 2 atributos que serão plotados
		for centroid in model.centroids.values():
			color  = centroid['color']
			data_x = []
			data_y = []
			for instance in centroid['instances']:
				data_x.append(instance[xi])
				data_y.append(instance[yi])
			# Coloca pontos do cluster no gráfico
			plt.scatter(data_x, data_y, color=color, marker='o')
			plt.xlabel(x_label)
			plt.ylabel(y_label)
			plt.grid()
		# Plota gráfico para os 2 atributos
		plt.savefig(f"../img/analysis/{x_label}_{y_label}.png")