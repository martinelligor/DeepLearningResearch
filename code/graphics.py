'''
    Author: Igor Martinelli - Undergraduate in Computer Science.

    This code is part of my cientific initiation in Institute of Mathematics and Computer Science (ICMC) - University of Sao Paulo.
    This code is responsible for build graphics of unbalanced data.
'''
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt

path = r'../datasets/'
datasets = glob.glob(path + "*.csv")
nrows = 6
ncols = 3
index = 1

for data in datasets:
    filename = data.split(path)
    dataset = pd.read_csv(data, index_col=0)
    y = dataset.iloc[:, -1].values

    plt.figure(1)
    plt.subplot(nrows, ncols, index)
    plt.plot(y, 'ro', color='b')
    plt.title(filename[1])
    index+=1

fig = plt.figure(1)
fig.savefig('dataset.png')
plt.show()
