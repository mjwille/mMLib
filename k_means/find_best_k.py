"""
Criado por: Marcelo Jantsch Wille
Email: marcelojantschwille@gmail.com
Última modificação: 03/11/2021
Descrição: Avaliação de qual o melhor 'k' para o algoritmo k-means.
"""

# Módulos de Python
from csv import reader
from copy import deepcopy
from math import inf
import matplotlib.pyplot as plt

# Módulos do projeto
from k_means import K_means

# Arquivo com dados
DATA_PATH  = "./data/bank_t2.csv"
# Booleano que indica se arquivo possui cabeçalho
HAS_HEADER = True

MAX_K = 20 # Valor máximo de 'k' para tentar encontrar o 'k' ideal
I = 100    # Número de iterações para cálculo da menor distância intracluster de certo 'k'


def get_lowest_wss(k, original_data):
	"""
	Devido à inicialização aleatória dos centróides, roda
	o modelo um certo número de vezes para um determinado valor
	de 'k'. Retorna a menor distância intracluster encontrada
	dentre todas essas iterações com esse 'k'.
	"""

	lowest_wss = inf

	# Gera modelo 'i' vezes para o valor de 'k' e calcula distâncias intracluster
	for _ in range(I):
		data = deepcopy(original_data)
		model = K_means(k, data)
		model.run()
		wss = model.get_wss()
		if wss < lowest_wss:
			lowest_wss = wss

	return lowest_wss   # menor distância intracluster encontrada para esse valor de 'k'


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

	k_values = [k for k in range(1, MAX_K+1)]

	# Roda k-means para diferentes valores de k
	wss_values = []
	for k in k_values:
		# Para cada valor de 'k', roda várias vezes o algoritmo
		wss = get_lowest_wss(k, data)
		wss_values.append(wss)
		print(f"k={k} has wss={wss}")

	# Plota resultados para então verificar melhor 'k' com método do cotovelo
	plt.figure()
	plt.plot(k_values, wss_values)
	plt.xticks(k_values)
	plt.xlabel("k")
	plt.ylabel("Dissimilaridade Intracluster")
	plt.grid()
	plt.title("Escolha do Número \"Ótimo\" de Clusters")

	# Mostra todos os gráficos gerados durante a execução
	plt.show()