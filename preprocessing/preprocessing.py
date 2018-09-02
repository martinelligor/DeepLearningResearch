'''
    Author: Igor Martinelli - Undergraduate in Computer Science.

    This code is a preprocessing part of my cientific initiation in Institute of Mathematics and Computer Science (ICMC) - University of Sao Paulo.
'''
# -*- coding: utf-8 -*-
import pandas as pd
import glob
from scipy import stats
from sklearn.preprocessing import LabelEncoder

#Taking name of path where the datasets is located.
path = r'../datasets/'
datasets = glob.glob(path + "*.txt")
data = []
#Opening datasets and preprocessing them.
for data in datasets:
    filename = data.split(path)
    #Reading datasets.
    dataset = pd.read_csv(filename[1], delimiter=' ')
    #Changing the result labels from NP and TP to 0 and 1.
    labelencoder_y = LabelEncoder()
    dataset.iloc[:, -1] = labelencoder_y.fit_transform(dataset.iloc[:, -1])
    #Computing the z-score for standardization, excluding the last column
    dataset.iloc[:, :-1] = stats.zscore(dataset.iloc[:, :-1])
    dataset.to_csv(filename[1][0:-3]+"csv")
