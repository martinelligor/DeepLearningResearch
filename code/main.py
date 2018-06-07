import numpy as np
import pandas as pd
import glob

path = r'../IC/Datasets/RNASeq/'
datasets = glob.glob(path + "*.csv")
data = []
balancing_methods = ["RandomOverSampler", "TomekLinks", "SMOTEENN",  "SMOTETomek", "EditedNearestNeighbours"]
dimensionality_reduction_methods = ["ReliefF", "LDA", "PCA", "KernelPCA"]
classification_methods = ["KNN", "LogisticRegression"]
number_of_features = [50, 100, 200]

for data in datasets:
	filename = data.split(path)
	dataset = pd.read_csv(data, index_col=0)


	X = dataset.iloc[:, :-1].values
	y = dataset.iloc[:, -1].values

	from sklearn.model_selection import StratifiedKFold
	from balancing_data import balancing_data
	from dimensionality_reduction import dimensionality_reduction
	from classification import classifier_model
	skf = StratifiedKFold(n_splits = 10, shuffle=True, random_state=0)
	accs = []
	cms = []
	scores = []
	means = []

	for balancing_method in balancing_methods:
		for classification_method in classification_methods:
			for dimensionality_reduction_method in dimensionality_reduction_methods:
				for num_features in number_of_features:
					print(balancing_method + " + " + classification_method + " + " + dimensionality_reduction_method + " + " + str(num_features) + " features + " + filename[1])

					for i, (train_index, test_index) in enumerate(skf.split(X, y)):

						#COLOCANDO OS DADOS NAS VARIÁVEIS DE TREINO E TESTE
						from sklearn.model_selection import train_test_split
						X_train, X_test = X[train_index], X[test_index]
						y_train, y_test = y[train_index], y[test_index]
						#balanceamento de dados
						X_train, y_train = balancing_data(X, y, balancing_method)
						#redução de dimensionalidade
						X_train, X_test = dimensionality_reduction(X_train, X_test, y_train, num_features, dimensionality_reduction_method)


						#PREDIZENDO RESULTADOS
						y_pred = classifier_model(X_test, X_train, y_train, classification_method)

						#MEDINDO A ACURÁCIA DO ALGORITMO
						from sklearn.metrics import accuracy_score
						acc = accuracy_score(y_test, y_pred)
						accs.append(acc)
						from sklearn.metrics import confusion_matrix
						cm = confusion_matrix(y_test, y_pred)
						cms.append(cm)

					print("Acuracia: ",accs)
					print("Media: ",np.mean(accs))
					print("")
					accs.clear()
