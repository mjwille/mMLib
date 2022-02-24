"""
Criado por: Marcelo Jantsch Wille
Email: marcelojantschwille@gmail.com
Última modificação: 20/10/2021
Descrição: Gera árvore de decisão a partir dos dados. Salva imagem da árvore
em formato png no sistema de arquivos com nome 'decision_tree.gv.png', dentro da pasta 'img'.
"""

# Módulos de Python
from csv import reader

# Módulos próprios do projeto
from tree import DecisionTree


DATA_PATH = "../data/benchmark.csv"    # Caminho do arquivo com dados no sistema
HAS_SAMPLING = False                   # Booleano que indica se árvore fará amostragem de atributos
HAS_BOOTSTRAP = False                  # Booleano que indica se terá bootstrap
MAX_H = None                           # Altura máxima da árvore gerada


if __name__ == '__main__':

	data = []

	# Abre e lê dados do arquivo com o dataset
	with open(DATA_PATH, 'r') as fp:
		csv_reader = reader(fp, delimiter=',')
		for line in csv_reader:
			data.append(line)


	# Gera árvore e cria arquivo com imagem
	dt = DecisionTree(max_height = MAX_H,
							has_sampling = HAS_SAMPLING,
							has_bootstrap = HAS_BOOTSTRAP)
	dt.train(data)
	dt.print_tree()
	dt.take_photo("decision_tree")

	new_instances = [
		["Chuvoso", "Quente", "Alta", "Falso"],
		["Nublado", "Amena", "Baixa", "Falso"],
		["Ensolarado", "Amena", "Normal", "Falso"],
		["Chuvoso", "Fria", "Alta", "Verdadeiro"],
		["Nublado", "Fria", "Alta", "Verdadeiro"],
	]

	# Classifica novas instâncias
	for new_instance in new_instances:
		prediction = dt.fit(new_instance)
		print(new_instance)
		print(f"{new_instance[-1]} ---> {prediction}")