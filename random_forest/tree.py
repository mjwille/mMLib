"""
Criado por: Marcelo Jantsch Wille
Email: marcelojantschwille@gmail.com
Última modificação: 20/10/2021
Descrição: Implementação da Árvore de Decisão e da Floresta Aleatória.
"""

# Módulos de Python
import graphviz as gz
from math import log2, sqrt, ceil
from statistics import mode, mean
from random import choice


def bootstrap(data, p = 0.7):
	""" Mecanismo de amostragem com reposição para criação de conjuntos de treino e teste """

	# Conjuntos de treino (sorteados com reposição) e 'Out of Bag' (instâncias não sorteadas)
	train_sample = []
	oob_sample  = []

	# Sorteia porcentagem 'p' das instâncias para comporem conjunto de treino
	for _ in range(int(len(data) * p)):
		train_sample.append(choice(data))

	# Preenche 'Out of Bag' com quem não foi sorteado
	for entry in data:
		if entry not in train_sample:
			oob_sample.append(entry)

	return train_sample, oob_sample



def is_numeric(str_value):
	""" Verifica se valor pertence ao tipo float ou ao tipo int """
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



class DecisionTree:
	""" Contém métodos para geração da árvore e classificação de novas instâncias """

	def __init__(self, max_height = None, has_sampling = False, has_bootstrap = False):
		self.tree = {}                       # Estrutura da árvore de decisão
		self.root = "Root"                   # Nome do nodo raíz
		self.max_height = max_height         # Altura máxima da árvore
		self.has_sampling = has_sampling     # Booleano que indica se tem amostragem de m atributos
		self.has_bootstrap = has_bootstrap   # Booleano que indica se tem bootstrap
		self.boot_train_set = None           # Conjundo de dados de treino do bootstrap
		self.boot_test_set = None            # Conjunto de dados de teste do bootstrap
		self.target_attr = None              # Label do atributo-alvo
		self.target_mode = None              # Moda do atributo-alvo
		self.attr_info = {}                  # Informação do índice e conjunto de valores de cada atributo


	def create_node(self, node_name, attrs, entries):
		""" Cria nodo na árvore de decisão self.tree da instância """

		self.tree[node_name]              = {}       # Cria nodo
		self.tree[node_name]['label']     = None     # Label que vai ser preenchido com atributo depois
		self.tree[node_name]['print']     = None     # Nome do nodo que vai ser impresso
		self.tree[node_name]['avg']       = None     # Valor a ser preenchido se atributo for numérico (média)
		self.tree[node_name]['info_gain'] = None     # Ganho de informação no nodo
		self.tree[node_name]['attrs']     = attrs    # Atributos disponíveis para próxima ramificação
		self.tree[node_name]['entries']   = entries  # Todas as entradas disponíveis para próxima ramificação
		self.tree[node_name]['branches']  = {}       # Ramos para os quais este node irá apontar


	def train(self, data):
		""" Gera estrutura da árvode baseado nos dados de entrada """

		# Pega nomes dos atributos e seus respectivos índices
		headers = data[0]
		for i, header in enumerate(headers):
			self.attr_info[header] = {'index': i, 'value_set': set(), 'is_numeric': None}

		# Define atributo-alvo (precisa ser o último da lista dos dados)
		self.target_attr = headers[-1]

		# Pega dados (cada entrada na tabela), valores possíveis e tipo (numérico, categórico) de cada atributo
		entries = []
		targets = []
		for entry in data[1:]:
			entries.append(entry)
			targets.append(entry[-1])
			# Monta o conjunto de valores possíveis para cada atributo
			for header, value in zip(headers, entry):
				self.attr_info[header]['value_set'].add(value)
				# Define se atributo é numérico ou categórico caso ainda não tenha sido definido
				if not self.attr_info[header]['is_numeric']:
					self.attr_info[header]['is_numeric'] = is_numeric(value)

		# Define moda do atributo-alvo
		self.target_mode = mode(targets)

		# Se bootsrap, usa como entrada do algoritmo conjunto de treino do bootstrap
		if self.has_bootstrap:
			self.boot_train_set, self.boot_test_set = bootstrap(entries)
			entries = self.boot_train_set

		# Cria nodo raíz
		self.create_node(self.root,                 # Nome do nodo inicial é "Root"
							  headers[:len(headers)-1],  # Todos os atributos menos a coluna y
							  entries)                   # Todas as entradas de dados

		# Gera estrutura da árvore recursivamente fazendo as ramificações com ganho de informação
		self.branch_out(self.tree[self.root], self.root, height = 1)


	def branch_out(self, node, node_name, height):
		""" Recursivamente faz ramificações na estrutura da árvore """

		# Pega todos os valores do atributo-alvo para testar critério de parada e calcular entropia
		target_values = []
		for values in node['entries']:
			target_values.append(values[-1])

		# Verifica critérios de parada da recursão
		if self.stop_branching(node, target_values, height):
			return

		# Calcula entropia do atributo-alvo
		target_entropy = DecisionTree.get_target_entropy(target_values)

		# Amostragem dos atributos
		sampled_attrs = self.sample_attrs(node['attrs'])

		# Decide próximo atributo pelo ganho de informação
		next_attr, info_gain = self.get_next_attr(sampled_attrs, node['entries'], target_entropy)

		# Coloca nome do atributo que tem maior ganho de informação no 'label' do nodo
		node['label'] = next_attr
		# Coloca ganho de informação do novo nodo
		node['info_gain'] = info_gain

		# Recursivamente ramifica para próximo nodo: ------------------------------

		# Se for numérico, abre ramos de menores ou iguais e de maiores
		if self.attr_info[next_attr]['is_numeric']:
			# Altera informação de 'print' e média do nodo
			attr_i = self.attr_info[next_attr]['index']
			avg_value = DecisionTree.get_attr_avg(attr_i, node['entries'])
			node['print'] = f"Attr {next_attr} ({avg_value:.2f})"
			node['avg'] = avg_value
			# Cria novo nodo para situação 'menor ou igual' e 'maior'
			for attr_value in ['Menor_Igual', 'Maior']:
				new_node_name = node_name + f"_{attr_value}_{avg_value:.2f}"
				attrs = [attr for attr in node['attrs'] if attr != next_attr]
				if attr_value == 'Menor_Igual':
					new_entries = DecisionTree.get_less_equal_entries(avg_value, attr_i, node['entries'])
				else:
					new_entries = DecisionTree.get_bigger_entries(avg_value, attr_i, node['entries'])
				self.create_node(new_node_name, attrs, new_entries)

				# Cria branch do atributo de maior ganho de informação a partir do nodo atual
				node['branches'][attr_value] = new_node_name
				# Recursão para o nodo criado
				self.branch_out(self.tree[new_node_name], new_node_name, height+1)

		# Se for categórico, abre ramo para cada categoria (cada valor possível do atributo)
		else:
			# Altera informação de 'print' do nodo
			node['print'] = next_attr
			# Cria novo nodo para cada um desses valores possíveis do atributo com maior ganho
			attr_values = self.attr_info[next_attr]['value_set']
			for attr_value in attr_values:
				new_node_name = node_name + f"_{attr_value}"
				attrs = [attr for attr in node['attrs'] if attr != next_attr]
				new_entries = self.get_entries_with_value(next_attr, attr_value, node['entries'])
				self.create_node(new_node_name, attrs, new_entries)

				# Cria branch do atributo de maior ganho de informação a partir do nodo atual
				node['branches'][attr_value] = new_node_name

				# Recursão para o nodo criado
				self.branch_out(self.tree[new_node_name], new_node_name, height+1)


	def stop_branching(self, node, target_values, height):
		""" Critérios de parada para ramificação da árvore de decisão """

		# Não tem mais entradas de dados pra calcular ganho de informação
		if len(node['entries']) == 0:
			node['label'] = f"{self.target_mode}"   # Folha é a moda de y no dataset completo
			node['print'] = f"{self.target_mode}"   # Folha é a moda de y no dataset completo
			return True

		# Não tem mais atributos para ramificar
		if len(node['attrs']) == 0:
			# Nó folha vai ser o valor de y mais frequente dos valores que restaram
			prediction = max(set(target_values), key=target_values.count)
			node['label'] = prediction
			node['print'] = prediction
			return True

		# Contém apenas um valor nos atributos-alvo
		if len(set(target_values)) == 1:
			prediction = target_values[0]
			node['label'] = prediction
			node['print'] = prediction
			return True

		# PODA: se altura máxima foi setada na criação do objeto, limita altura da árvore
		if self.max_height:
			if height == self.max_height:
				# Valor predito é aquele mais frequente dentre os valores de y que chegaram no nodo
				prediction = max(set(target_values), key=target_values.count)
				node['label'] = prediction
				node['print'] = prediction
				return True

		return False   # Continua fazendo as ramificações


	def sample_attrs(self, attrs_list):
		""" Amostragem dos atributos a cada divisão do nó (raiz quadrada dos m atributos) """

		# Se não tem amostragem, retorna a própria lista de atributos do nodo
		if not self.has_sampling:
			return attrs_list

		# Total de atributos amostrados é raíz quadrada do número de atributos
		m = ceil(sqrt(len(attrs_list)))

		# Escolhe aleatoriamente 'm' atributos
		sampled_attrs = []
		for _ in range(m):
			sampled_attrs.append(choice(attrs_list))

		return sampled_attrs


	def get_next_attr(self, attrs, entries, target_entropy):
		""" Decide próximo atributo baseado na entropia e ganho de informação """

		# Calculo da entropia necessita todos os valores possíveis do atributo-alvo
		target_values = self.attr_info[self.target_attr]['value_set']

		# Para cada atributo disponível para ramificar, calcula o ganho de informação
		next_attr = {'attr': None, 'info_gain': -1}
		for attr in attrs:
			# Pega metainformação do atributo
			attr_i = self.attr_info[attr]['index']
			attr_values = self.attr_info[attr]['value_set']

			# Calcula a entropia do atributo para as entradas de dados disponíveis no nodo
			if self.attr_info[attr]['is_numeric']:
				attr_entropy = DecisionTree.get_numerical_entropy(attr_i,
																				  target_values,
																				  entries)
			else:
				attr_entropy = DecisionTree.get_categorical_entropy(attr_i,
																					 attr_values,
																					 target_values,
																					 entries)

			# Verifica se ganho de informação é maior do que o atributo com maior ganho no momento
			info_gain = DecisionTree.info_gain(target_entropy, attr_entropy)
			if info_gain > next_attr['info_gain']:
				next_attr['attr'] = attr
				next_attr['info_gain'] = info_gain

		return next_attr['attr'], next_attr['info_gain']  # Atributo com maior ganho de informação


	@staticmethod
	def get_target_entropy(target_values):
		""" Calcula a entropia dos atributos-alvo y """
		entropy = 0
		for value in set(target_values):
			p = target_values.count(value) / len(target_values)
			entropy += -p*log2(p)
		return entropy


	@staticmethod
	def get_categorical_entropy(attr_index, attr_values, target_values, entries):
		""" Calcula a entropia de um atributo categórico do vetor X """

		# Preenche estrutura que associa cada valor possível do atributo X aos valores possíveis do alvo
		# {
		#   'Ensolarado': {'Sim': 0, 'Não': 0},
		#   'Nublado':    {'Sim': 0, 'Não': 0},
		#   'Chuvoso':    {'Sim': 0, 'Não': 0},
		# }
		amounts = {}
		for attr_value in attr_values:
			amounts[attr_value] = {}
			for target_value in target_values:
				amounts[attr_value][target_value] = 0

		# Conta cada caso do atributo-alvo para cada valor possível do atributo
		for entry in entries:
			attr_value = entry[attr_index]
			target_value = entry[-1]
			amounts[attr_value][target_value] += 1

		# Faz calculo da entropia para cada valor possível do atributo X
		entropy_values = []
		for attr_value in attr_values:
			total = 0
			for value in amounts[attr_value].values():
				total += value

			entropy = 0
			for target_value in target_values:
				if amounts[attr_value][target_value] != 0:
					p = amounts[attr_value][target_value] / total
					entropy += -p*log2(p)

			entropy_values.append([entropy, total])

		# Calcula entropia média da entropia de cada valor possível do atributo X
		entropy_avg = 0
		for entropy, total in entropy_values:
			entropy_avg += (total/len(entries)) * entropy

		return entropy_avg    # Entropia média


	@staticmethod
	def get_numerical_entropy(attr_index, target_values, entries):
		""" Calcula entropia de um atributo numérico do vetor X"""

		# Pega valor médio do atributo para o conjunto de dados que chegou naquele nodo
		avg_attr_value = DecisionTree.get_attr_avg(attr_index, entries)

		# Preenche estrutura que associa cada categoria numérica em relação à média a um valor possível do atributo-alvo
		# {
		#     'Menor_Igual': {'Type 1': 0, 'Type 2': 0, 'Type 3': 0}
		#     'Maior'      : {'Type 1': 0, 'Type 2': 0, 'Type 3': 0}
		# }
		amounts = {}
		amounts['Menor_Igual'] = {}
		amounts['Maior']       = {}
		for target_value in target_values:
			amounts['Menor_Igual'][target_value] = 0
			amounts['Maior'][target_value]       = 0

		# Conta cada caso das entradas, preenchendo a estrutura definida acima
		for entry in entries:
			attr_value = float(entry[attr_index])
			target_value = entry[-1]
			if attr_value <= avg_attr_value:
				amounts['Menor_Igual'][target_value] += 1
			else:
				amounts['Maior'][target_value] += 1

		# Faz calculo da entropia
		entropy_values = []
		for attr_value in ['Menor_Igual', 'Maior']:
			total = 0
			for value in amounts[attr_value].values():
				total += value

			entropy = 0
			for target_value in target_values:
				if amounts[attr_value][target_value] != 0:
					p = amounts[attr_value][target_value] / total
					entropy += -p*log2(p)

			entropy_values.append([entropy, total])

		# Calcula entropia média da entropia de cada categoria numérica (menor_igual ou maior) do atributo X
		entropy_avg = 0
		for entropy, total in entropy_values:
			entropy_avg += (total/len(entries)) * entropy

		return entropy_avg   #Entropia média


	@staticmethod
	def info_gain(target_entropy, attr_entropy):
		""" Calcula o ganho de informação dadas as entropias """
		return target_entropy - attr_entropy


	def get_entries_with_value(self, attr, value, entries):
		""" Pega entradas que possuem certo valor em um atributo """

		# Pega metadados do atributo para acessar os dados
		attr_i = self.attr_info[attr]['index']

		# Para cada entrada, pega somente aquelas com o valor do atributo passado como argumento
		entries_with_value = []
		for entry in entries:
			if entry[attr_i] == value:
				entries_with_value.append(entry)
		return entries_with_value


	@staticmethod
	def get_less_equal_entries(value, attr_i, entries):
		""" Pega todas as instânias com valor de certo atributo menor ou igual que um valor """
		new_entries = []
		for entry in entries:
			if float(entry[attr_i]) <= value:
				new_entries.append(entry)
		return new_entries


	@staticmethod
	def get_bigger_entries(value, attr_i, entries):
		""" Pega todas as instânias com valor de certo atributo maior que um valor """
		new_entries = []
		for entry in entries:
			if float(entry[attr_i]) > value:
				new_entries.append(entry)
		return new_entries


	@staticmethod
	def get_attr_avg(attr_index, entries):
		""" Retorna valor médio para atributo com certo índice nos dados """
		attr_values = []
		for entry in entries:
			attr_values.append(float(entry[attr_index]))

		return mean(attr_values)


	def fit(self, instance):
		""" Faz classificação de nova instância percorrendo a árvore """

		# Nodo inicial é o "Root"
		node = self.tree[self.root]
		value = node['label']

		# Enquanto não chegou no nodo folha
		while node['branches']:
			# Pega valor do nodo (atributo ou resposta) na instância passada
			attr_i = self.attr_info[value]['index']
			attr_value =  instance[attr_i]

			if self.attr_info[value]['is_numeric']:
				if float(attr_value) <= node['avg']:
					node = self.tree[node['branches']['Menor_Igual']]
				else:
					node = self.tree[node['branches']['Maior']]
			else:
				# Agora que tem o valor, vai para o próximo nodo na árvore
				node = self.tree[node['branches'][attr_value]]

			value = node['label']

		return value   # Resposta da classificação


	def fit_bootstrap(self, debug = False):
		""" Faz classificação de todas as instâncias de teste do bootstrap e retorna acurácia """

		if not self.has_bootstrap:
			raise Exception("The decision tree was created without bootstrap.")

		# Total de erros nas predições do conjunto de teste do bootstrap
		errors = 0

		# Faz 'fit' de cada instância do conjunto de teste do bootstrap
		for instance in self.boot_test_set:
			expected_prediction = instance[-1]
			prediction = self.fit(instance)

			if debug:
				print(f"{instance}\n{expected_prediction} ---> {prediction}")

			if prediction != expected_prediction:
				errors += 1

		accuracy = (1 - (errors / len(self.boot_test_set))) * 100

		if debug:
			print(f"Accuracy = {accuracy:.2f}%")

		return accuracy


	def print_tree(self):
		""" Imprime árvore de decisão gerada no terminal """
		print("------------------ ÁRVORE INÍCIO ------------------")
		self.recursive_print(self.tree[self.root])
		print("------------------- ÁRVORE FIM --------------------")


	def recursive_print(self, node, level = 0):
		""" Recursivamente imprime a árvore de decisão nodo a nodo """

		# Imprime valor do nodo atual com seu nível de identação de acordo com sua profundidade
		for _ in range(level):
			print("  ", end="")

		# Testa se tem ganho de informação (folha) para não imprimir ganho de informação
		if node['info_gain']:
			print(f"{node['print']} [{node['info_gain']:.3f}]")
		else:
			print(f"{node['print']}")

		# Recursivamente imprime os filhos do nodo
		for branchLabel, son in node['branches'].items():
			self.recursive_print(self.tree[son], level+1)


	def take_photo(self, filename):
		""" Coloca em arquivo de saída a imagem da árvore """
		tree_img = gz.Digraph(format = 'png')
		self.recursive_photo(self.tree[self.root], tree_img)
		tree_img.render(f"../img/{filename}.gv")


	def recursive_photo(self, node, tree_img, i = 0):
		""" Recursivamente passa pela árvore, pegando os nodos e arestas para gerar imagem """

		# Coloca nodo na imagem final (testa se tem ganho de informação, isto é, se não é folha)
		if node['info_gain']:
			node_str = f"{i}. {node['print']}\n[{node['info_gain']:.3f}]"
		else:
			node_str = f"{i}. {node['print']}"


		# Se nodo folha, coloca estilo diferente (cor e forma)
		if not node['branches']:
			tree_img.node(node_str, color = 'red', shape = 'box', fontcolor = 'red')
		else:
			tree_img.node(node_str)

		# Para cada filho
		for branchLabel, son in node['branches'].items():
			# Coloca aresta entre o nodo e seu filho com o atributo que levou àquele nodo filho
			i += 1

			# Testa se tem ganho de informação (folha) para não imprimir ganho de informação
			if self.tree[son]['info_gain']:
				son_str = f"{i}. {self.tree[son]['print']}\n[{self.tree[son]['info_gain']:.3f}]"
			else:
				son_str = f"{i}. {self.tree[son]['print']}"

			tree_img.edge(node_str, son_str, label=branchLabel)

			# Coloca também na imagem filhos do filho
			i = self.recursive_photo(self.tree[son], tree_img, i)

		return i    # Retorna variável de controle 'i' usada para não repetir nomes dos nodos



class Ensemble:
	""" Ensemble de 'n' árvores de decisão (Floresta Aleatória) """

	def __init__(self, ntree, max_height = None, id = 0):
		self.number_of_trees = ntree   # Número de árvores da floresta
		self.max_height = max_height   # Altura máxima de todas as árvores da floresta
		self.ensemble_id = id          # Id para gerar pasta com imagens das árvores da floresta
		self.decision_trees = []       # Lista com as árvores


	def generate(self, data, get_tree_images = False):
		""" Gera árvores para os dados (fold se for validação cruzada estratificada) """
		for i in range(self.number_of_trees):
			decision_tree = DecisionTree(max_height = self.max_height,
												  has_sampling = True,
												  has_bootstrap = True)
			decision_tree.train(data)

			if get_tree_images:
				decision_tree.take_photo(f"Ensemble_{self.ensemble_id}/tree_{i}")

			self.decision_trees.append(decision_tree)


	def fit(self, instance):
		""" Faz classificação de nova instância na floresta aleatória """

		# Predições de todas as árvores da floresta
		predictions = []

		# Cada árvore faz sua predição
		for decision_tree in self.decision_trees:
			predictions.append(decision_tree.fit(instance))

		# Combina todas as predições de cada árvore em uma única predição
		prediction = self.combine(predictions)
		return prediction


	def combine(self, predictions):
		""" Combinação por votação majoritária das classificações das árvores do ensemble """
		return max(set(predictions), key=predictions.count)