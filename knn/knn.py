"""
Criado por: Marcelo Jantsch Wille
Última mudança: 16/08/2021
Email: marcelojantschwille@gmail.com
Descrição:
	Implementação do algoritmo KNN com opção
	de alterar o hyperparâmetro 'k' e definir
	qual a distância utilizada. Os dados também
	são divididos pelo método Holdout em
	treinamento e teste.
"""

import csv
from sys import argv
from math import sqrt
from collections import Counter

DATA_PATH = './breast_cancer.csv'
NORMALIZED_PATH = './normalized.csv'
NORMALIZED_KNN = False
K_VALUE = 0
DISTANCE_TYPE = 'euclidian'


def get_args():
	"""
	Processa argumentos passados na chamada do programa
	e altera variáveis globais baseado nos parâmetros passados.
	"""
	global NORMALIZED_KNN
	global K_VALUE
	global DISTANCE_TYPE
	if '-k' in argv:
		K_VALUE = argv[argv.index('-k')+1]
		try:
			K_VALUE = int(K_VALUE)
			if K_VALUE <= 0:
				raise ValueError
		except ValueError:
			print("Erro! 'k' deve ser natural positivo.")
			exit(-1)
	else:
		print("Erro! 'k' não foi informado.\nUso: python3 knn.py -k <V>.")
		print("Digite 'python3 knn.py -h' para obter ajuda.")
		exit(-1)

	if '-d' in argv:
		DISTANCE_TYPE = argv[argv.index('-d')+1]
		if (DISTANCE_TYPE != 'euclidian') and (DISTANCE_TYPE != 'manhattan'):
			print("Erro! Distância deve ser um dos valores: 'euclidian' ou 'manhattan'.")
			exit(-1)

	if '-n' in argv:
		NORMALIZED_KNN = True

	if '-h' in argv:
		print_help()


def print_help():
	"""
	Função que imprime no terminal as possíveis flags do programa.
	"""
	print("-k <V> para alterar o valor de 'k', que deve ser natural positivo.")
	print("-d <euclidian | manhattan> para alterar o algoritmo de distância.")
	print("-n para usar os dados normalizados.")
	print("-h para ajuda com as flags do comando.")


def print_parameters():
	"""
	Imprime na tela os parâmetro usados na execução do programa
	para o valor do 'k' e o algoritmo da distância utilizado.
	"""
	global K_VALUE
	global DISTANCE_TYPE

	print(f"k={K_VALUE}  distance={DISTANCE_TYPE} ===>", end=" ")


def holdout(fp, prop_training = 0.8):
	"""
	Aplica o método Holdout para separar dados disponíveis em
	dados de treinamento e dados de treino.
	:param fp: File pointer do arquivo com os dados.
	:param prop_training: Proporção de dados que serão de treinamento.
	:return: 2 listas referentes aos dados de treinamento e de teste.
	"""

	data = csv.DictReader(fp)

	targets = {}
	for instance in data:
		target = instance['target']
		targets[target] = targets.get(target, 0) + 1

	for target in targets.keys():
		targets[target] = int(targets[target] * prop_training)

	fp.seek(0)
	data = csv.DictReader(fp)

	training_data = []
	test_data = []

	for instance in data:
		target = instance['target']
		instance.pop('N')
		if targets[target] != 0:
			targets[target] -= 1
			training_data.append(instance)
		else:
			test_data.append(instance)

	return training_data, test_data


def calculate_distance(xi, xj):
	"""
	Calcula distância entre 2 instâncias. O algoritmo da
	distância é decidido baseado na variável global distance_type.
	:param xi: Instância de dados de teste.
	:param xj: Instância de dados de treinamento.
	:return: Valor real da distância entre as duas instâncias,
	calculado atributo por atributo.
	"""
	global DISTANCE_TYPE
	distance = 0

	if DISTANCE_TYPE == 'euclidian':
		for attr_xi, attr_xj in zip(xi,xj):
			distance += (float(xi[attr_xi]) - float(xj[attr_xj]))**2
		distance = sqrt(distance)

	elif DISTANCE_TYPE == 'manhattan':
		for attr_xi, attr_xj in zip(xi, xj):
			distance += abs(float(xi[attr_xi]) - float(xj[attr_xj]))

	return distance


def training_data_distance(xi, training_data):
	"""
	Coloca como atributo de cada instância dos dados de
	treinamento a distância em relação à instância dos
	dados de teste 'xi'.
	:param xi: Instância dos dados de teste.
	:param training_data: Dados de treinamento.
	"""
	for xj in training_data:
		distance = calculate_distance(xi, xj)
		xj['distance'] = distance


def get_closest_ks(training_data):
	"""
	Função que retorna as 'k' instâncias mais próximas da
	última instância processada, pois valores de distância
	vão estar no atributo 'distance', que foi criado nos
	dados de treinamento.
	:param training_data: Dados de treinamento.
	:return: 'k' instâncias com menor valor de distância.
	"""
	global K_VALUE
	sorted_training_data = sorted(training_data, key=lambda l: l['distance'])
	return sorted_training_data[:K_VALUE]


def get_most_common_target(instance_set):
	"""
	Retorna o atributo alvo mais comum presente no
	conjunto de instâncias passado como parâmetro.
	:param instance_set: Conjunto de instâncias ('k' mais próximos).
	:return: O valor mais comum dos atributos alvo do conjunto de instâncias.
	"""
	target_counter = Counter(instance['target'] for instance in instance_set)
	return target_counter.most_common(1)[0][0]


def calculate_accuracy(test_data):
	"""
	Calcula a acurácia dos resultados da aplicação do KNN,
	comparando o atributo alvo com o valor previsto pelo algoritmo.
	:param test_data: Dados de teste.
	:return: Valor da acurácia para todas as instâncias de teste.
	"""
	errors = 0
	for instance in test_data:
		if instance['target'] != instance['prediction']:
			errors += 1

	errors = errors / len(test_data) * 100
	accuracy = 100 - errors
	return accuracy


def run(k, distance_type, normalized = False):
	"""
	Roda o algoritmo KNN.
	:return: Acurácia do algoritmo para todas as instâncias de teste.
	"""
	global DATA_PATH
	global NORMALIZED_PATH
	global NORMALIZED_KNN
	global K_VALUE
	global DISTANCE_TYPE

	K_VALUE = k
	DISTANCE_TYPE = distance_type
	NORMALIZED_KNN = normalized

	print_parameters()

	if NORMALIZED_KNN:
		filepath = NORMALIZED_PATH
	else:
		filepath = DATA_PATH

	with open(filepath) as fp:
		training_data, test_data = holdout(fp)
		for xi in test_data:
			training_data_distance(xi, training_data)
			closest_ks = get_closest_ks(training_data)
			most_common_target = get_most_common_target(closest_ks)
			xi['prediction'] = most_common_target

		accuracy =  calculate_accuracy(test_data)
		print(f"Accuracy: {accuracy:.2f}%")
		return accuracy


if __name__ == '__main__':
	get_args()
	run(K_VALUE, DISTANCE_TYPE, NORMALIZED_KNN)
