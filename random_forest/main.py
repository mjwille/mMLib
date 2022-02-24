"""
Criado por: Marcelo Jantsch Wille
Email: marcelojantschwille@gmail.com
Última modificação: 20/10/2021
Descrição: Avaliação do desempenho do modelo usando cross-validation.
Gera imagem do gráfico com as acurácias para diferentes valores de ntree.
"""

# Módulos de Python
import matplotlib.pyplot as plt
from csv import reader

# Módulos próprios do projeto
from validation import cross_validation


DATA_PATH = "../data/house_votes_84.csv"   # Caminho do arquivo com dados no sistema
CROSS_K   = 10                             # Valor do K na validação cruzada
MAX_H     = None                           # Altura máxima das árvores na floresta



if __name__ == '__main__':

	data = []

	# Abre e lê dados do arquivo com o dataset
	with open(DATA_PATH, 'r') as fp:
		csv_reader = reader(fp, delimiter=',')
		for line in csv_reader:
			data.append(line)


	# Avalia acurácia da validação cruzada para diferentes quantidades de árvores

	ntrees = [n for n in range(5, 101, 5)]

	accuracies = []
	for ntree in ntrees:
		accuracy, stdev = cross_validation(data, ntree, K = CROSS_K, max_height = MAX_H)
		accuracies.append(accuracy)
		print(f"ntree = {ntree} ------> accuracy = {accuracy:.2f}%")

	# Mostra gráfico com acurácias
	plt.plot(ntrees, accuracies, label = "Acurácia", marker = 'o')
	plt.xlabel('Número de Árvores')
	plt.ylabel('Acurácia')
	plt.grid()
	plt.title('Impacto do Número de Árvores')
	plt.legend()
	plt.show()