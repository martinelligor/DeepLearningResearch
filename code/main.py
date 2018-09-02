'''
    Author: Igor Martinelli - Undergraduate in Computer Science.

    This code is part of my cientific initiation in Institute of Mathematics and Computer Science (ICMC) - University of Sao Paulo.
    This code is the pipeline that is responsible for all the experiments happened
'''
# -*- coding: utf-8 -*-
import glob
import numpy as np
import pandas as pd
from balancing_data import balancing_data
from sklearn.metrics import accuracy_score
from classification import classifier_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from dimensionality_reduction import dimensionality_reduction

path = r'../datasets/'
datasets = glob.glob(path + "*.csv")
data = []
balancing_methods = ["RandomOverSampler", "TomekLinks", "SMOTEENN",  "SMOTETomek", "EditedNearestNeighbours"]
dimensionality_reduction_methods = ["ReliefF", "PCA"]
number_of_features = [10, 50, 100, 200]
classification_methods = ["KNN", "LogisticRegression", "DecisionTree", "MLP_2_Sigmoid", "MLP_2_Tanh", "MLP_3"]

for data in datasets:
    filename = data.split(path)
    dataset = pd.read_csv(data, index_col=0)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    skf = StratifiedKFold(n_splits = 10, shuffle=True, random_state=0)
    accs = []
    cms = []
    scores = []
    means = []

    for balancing_method in balancing_methods:
        for classification_method in classification_methods:
            for dimensionality_reduction_method in dimensionality_reduction_methods:
                for num_features in number_of_features:
                    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
                        #valores nas variaveis de treino e teste
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        #balanceamento de dados
                        X_train, y_train = balancing_data(X, y, balancing_method)
                        #redução de dimensionalidade
                        X_train, X_test = dimensionality_reduction(X_train, X_test, y_train, num_features, dimensionality_reduction_method)

                        #PREDIZENDO RESULTADOS
                        y_pred = classifier_model(X_test, X_train, y_train, classification_method)
                        #MEDINDO A ACURÁCIA DO ALGORITMO
                        acc = accuracy_score(y_test, y_pred)
                        accs.append(acc)
    
                    result_file = open("../results/" + filename[1].split(".csv")[0] + ".txt", "a")
                    result_file.write(filename[1] + " + " + balancing_method + " + " + classification_method + " + " + dimensionality_reduction_method + " + " + str(num_features) + " features ," + str(np.mean(accs)) + "\n")
                    result_file.close()

                    accs.clear()