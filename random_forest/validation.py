"""
Criado por: Marcelo Jantsch Wille
Email: marcelojantschwille@gmail.com
Última modificação: 20/10/2021
Descrição: Implementações das funções relevantes para validar
a generalização do modelo: validação cruzada estratificada.
"""

# Módulos de Python
from statistics import mean, stdev

# Módulos próprios do projeto
from tree import Ensemble



def cross_validation(data, ntree, K = 10, max_height = None, get_tree_images = False):
	""" Treina floresta para os k-1 folds e testa no fold que sobrou. Repete k vezes e pega média """

	headers = data[0]

	# Medidas de desempenho (acurácia) de cada validação cruzada
	accuracies = []

	# Faz divisão dos dados em 'folds' estratificados (mesma proporção do target)
	folds = get_folds(K, data[1:])

	# A cada iteração, um fold será pra teste e o resto para treino
	for i in range(K):
		# Fold 'i' será o fold de teste da acurácia
		test_data  = folds[i]

		# O resto dos folds é pra treino da floresta, e precisam ser juntados
		train_folds = folds[:i] + folds[i+1:]
		train_data = glue_folds_together(headers, train_folds)

		# Treina as árvores da floresta com folds de treino
		randomForest = Ensemble(ntree, max_height, i)
		randomForest.generate(train_data, get_tree_images)

		# Tenta prever com a floresta treina cada instância do fold de teste
		errors = 0
		for instance in test_data:
			prediction = randomForest.fit(instance)
			if prediction != instance[-1]:
				errors += 1

		# Coloca acurácia da floresta na lista com todas as acurácias
		accuracy = (1 - errors / len(test_data)) * 100
		accuracies.append(accuracy)

	# Medidas de desempenho final do modelo (média e desvio padrão)
	avg_accuracy = mean(accuracies)
	stdev_accuracy = stdev(accuracies)
	return avg_accuracy, stdev_accuracy



def get_folds(k, data):
	""" Faz estratificação dos dados, mantendo a mesma proporção das classes de target entre folds """

	folds = [[] for _ in range(k)]

	# Cada classe do atributo-alvo tem uma entrada no dicionário
	# {
	#   "Democrata": [todas as instâncias com Democrata como valor de y],
	#   "Republicano": [todas as instâncias com Republicano como valor de y],
	# }com uma lista de suas instâncias
	target_instances = {}

	# Preenche dicionário com as instâncias de cada atributo-alvo
	for entry in data:
		target_value = entry[-1]
		if target_value not in target_instances:
			target_instances[target_value] = []

		target_instances[target_value].append(entry)

	# Preenche folds com cada instância de cada classe do atributo-alvo
	# de forma rotativa (fold 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, ...)
	i = 0
	for target, instances in target_instances.items():
		for instance in instances:
			j = i % k
			folds[j].append(instance)
			i+=1

	return folds



def glue_folds_together(headers, folds):
	""" Coloca dados de todos os folds que serão usados pra treino e nomes dos atributos juntos """
	train_data = [headers]
	for fold in folds:
		for instance in fold:
			train_data.append(instance)
	return train_data