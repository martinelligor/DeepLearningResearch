'''
Author: Igor Martinelli - Undergraduate in Computer Science.

This code is part of my cientific initiation in Institute of Mathematics and Computer Science (ICMC) - University of Sao Paulo.
This code is responsible for classification
'''
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
# -*- coding: utf-8 -*-
def classifier_model(X_test, X_train, y_train, classifier_method):

    if classifier_method == "KNN":
        #RODANDO O ALGORITMO
        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

    elif classifier_method == "LogisticRegression":
        #Fitting Logistic Regression to the Training Set
        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

    elif classifier_method == "DecisionTree":
        classifier  = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
    
    elif classifier_method == "MLP_2_Sigmoid":
        y_pred = (MLP_2_Layers_Sigmoid(X_train, y_train, X_test))

    elif classifier_method == "MLP_2_Tanh":
        y_pred = (MLP_2_Layers_Tanh(X_train, y_train, X_test))
   
    elif classifier_method == "MLP_3":
        y_pred = (MLP_3_Layers(X_train, y_train, X_test))

    return y_pred

def MLP_2_Layers_Sigmoid(X_train, y_train, X_test):
    input_size = X_train.shape[1]
    model = Sequential()

    model.add(Dense(input_size,
        kernel_initializer = "glorot_uniform",
        bias_initializer = "glorot_uniform",
        activation = "sigmoid"))

    model.add(Dropout(0.1))

    model.add(Dense(input_size,
        kernel_initializer = "glorot_uniform",
        bias_initializer = "glorot_uniform",
        activation = "sigmoid"))

    model.add(Dropout(0.1))

    # Output layer consisting of just one node for classification
    model.add(Dense(1,
        kernel_initializer = "glorot_uniform",
        bias_initializer = "glorot_uniform",
        activation = "sigmoid",
        name = "output"))

    model.compile(loss = "binary_crossentropy",
            optimizer = "rmsprop",
            metrics = ["accuracy"])

    model.fit(X_train, y_train, epochs = 200, batch_size = 32)
    y_pred = np.around(np.asarray(model.predict(X_test, batch_size = 32)))

    return y_pred

def MLP_2_Layers_Tanh(X_train, y_train, X_test):
    input_size = X_train.shape[1]
    model = Sequential()

    model.add(Dense(input_size,
                            kernel_initializer = "glorot_uniform",
                            bias_initializer = "glorot_uniform",
                            activation = "tanh"))

    model.add(Dropout(0.1))

    model.add(Dense(input_size,
                            kernel_initializer = "glorot_uniform",
                            bias_initializer = "glorot_uniform",
                            activation = "tanh"))

    model.add(Dropout(0.1))

    # Output layer consisting of just one node for classification
    model.add(Dense(1,
                            kernel_initializer = "glorot_uniform",
                            bias_initializer = "glorot_uniform",
                            activation = "sigmoid",
                            name = "output"))

    model.compile(loss = "binary_crossentropy",
                    optimizer = "rmsprop",
                    metrics = ["accuracy"])

    model.fit(X_train, y_train, epochs = 200, batch_size = 32)
    y_pred = np.around(np.asarray(model.predict(X_test, batch_size = 32)))

    return y_pred

def MLP_3_Layers(X_train, y_train, X_test):
    input_size = X_train.shape[1]
    model = Sequential()

    model.add(Dense(input_size,
                            kernel_initializer = "glorot_uniform",
                            bias_initializer = "glorot_uniform",
                            activation = "tanh"))

    model.add(Dropout(0.1))

    model.add(Dense(input_size,
                            kernel_initializer = "glorot_uniform",
                            bias_initializer = "glorot_uniform",
                            activation = "tanh"))

    model.add(Dropout(0.1))

    model.add(Dense(input_size,
                            kernel_initializer = "glorot_uniform",
                            bias_initializer = "glorot_uniform",
                            activation = "tanh"))

    model.add(Dropout(0.1))

    # Output layer consisting of just one node for classification
    model.add(Dense(1,
                            kernel_initializer = "glorot_uniform",
                            bias_initializer = "glorot_uniform",
                            activation = "sigmoid",
                            name = "output"))

    model.compile(loss = "binary_crossentropy",
                    optimizer = "rmsprop",
                    metrics = ["accuracy"])

    model.fit(X_train, y_train, epochs = 200, batch_size = 32)
    y_pred = np.around(np.asarray(model.predict(X_test, batch_size = 32)))

    return y_pred

