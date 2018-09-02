'''
    Author: Igor Martinelli - Undergraduate in Computer Science.

    This code is part of my cientific initiation in Institute of Mathematics and Computer Science (ICMC) - University of Sao Paulo.
    This code is responsible for balancing data process.
'''
# -*- coding: utf-8 -*-
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter

#This function is responsible for balancing the dataset by a provided method.
def balancing_data(X, y, method):

    if method == "RandomOverSampler":
        b_method = RandomOverSampler(random_state=0)
    elif method == "TomekLinks":
        b_method = TomekLinks(random_state=0)
    elif method == "SMOTEENN":
        b_method = SMOTEENN(random_state=0)
    elif method == "SMOTETomek":
        b_method = SMOTETomek(random_state=0)
    elif method == "EditedNearestNeighbours":
        b_method = EditedNearestNeighbours(random_state = 0)

    #Balancing and returning the balanced data.
    X_resampled, y_resampled = b_method.fit_sample(X, y)

    return(X_resampled, y_resampled)
