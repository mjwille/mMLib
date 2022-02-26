import csv

# Atributos não normalizados com seus respectivos valores: [mínimo, máximo]
NOT_YET_NORMALIZED = {'X2' : [6.98, 28.11],
                        'X3' : [9.71, 39.28],
                        'X4' : [43.79, 188.50],
                        'X5' : [143.50, 2501],
                        'X14': [0.75, 21.98],
                        'X15': [6.80, 542.20],
                        'X22': [7.93, 36.04],
                        'X23': [12.02, 49.54],
                        'X24': [50.41, 251.20],
                        'X25': [185.20, 4254]}

if __name__ == '__main__':
	with open('./breast_cancer.csv') as fp1, open('./normalized.csv', 'w') as fp2:
		data = csv.DictReader(fp1)
		writer = csv.writer(fp2)

		header = [f'X{n}' for n in range(2, 32)]
		header.insert(0, 'N')
		header.append('target')
		writer.writerow(header)

		for instance in data:
			new_row = []
			for attr in instance:
				if attr in NOT_YET_NORMALIZED.keys():
					min_v = NOT_YET_NORMALIZED.get(attr)[0]
					max_v = NOT_YET_NORMALIZED.get(attr)[1]
					current_v = float(instance[attr])
					new_v = (current_v -  min_v) / (max_v - min_v)

					new_row.append(f"{new_v:.4f}")
				else:
					new_row.append(instance[attr])
			writer.writerow(new_row)