"""
Criado por: Marcelo Jantsch Wille
Email: marcelojantschwille@gmail.com
Última modificação: 20/10/2021
Descrição: Gera floresta aleatória com 5 árvores a partir dos dados. Imagens de cada
árvore gerada estão dentro da pasta 'Ensemble_0'.
"""

# Módulos de Python
from csv import reader

# Módulos próprios do projeto
from tree import Ensemble


DATA_PATH = "../data/wine_recognition.csv"   # Caminho do arquivo com dados no sistema
MAX_H = None                                 # Altura máxima da árvore gerada

if __name__ == '__main__':

	data = []

	# Abre e lê dados do arquivo com o dataset
	with open(DATA_PATH, 'r') as fp:
		csv_reader = reader(fp, delimiter=',')
		for line in csv_reader:
			data.append(line)

	# Supõe que última instância é uma nova instância apenas para testar que 'fit' funciona
	new_instance = data[-1]

	forest = Ensemble(5, max_height = MAX_H)
	forest.generate(data, get_tree_images = True)
	prediction = forest.fit(new_instance)

	print(f"{new_instance[-1]} ---> {prediction}")