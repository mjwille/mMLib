"""
Criado por: Marcelo Jantsch Wille
Email: marcelojantschwille@gmail.com
Última modificação: 04/11/2021
Descrição: Plota clusters gerados na execução do k-means.
"""

# Módulos de Python
from csv import reader
from copy import deepcopy
import matplotlib.pyplot as plt

# Módulos do projeto
from k_means import K_means

# Arquivo com dados
DATA_PATH  = "../data/benchmark_instances.csv"
# Booleano que indica se arquivo possui cabeçalho
HAS_HEADER = False
# Arquivo onde o gráfico dos clusters será salvo
OUTPUT_FILE = "../img/clusters.png"

K = 15    # Valor de 'K' com o qual o algoritmo K-means será executado
I = 100   # Número de iterações para cálculo da menor distância intracluster de certo 'k'

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

	# Eliminar headers, pois objeto K_means exige dados passados sem eles
	if HAS_HEADER:
		data = data[1:]

	# Executa K-means
	model = get_lowest_wss_model(K, data)

	# Salva gráfico cos clusters encontrados
	model.plot_clusters()
	plt.savefig(OUTPUT_FILE)
	plt.show()