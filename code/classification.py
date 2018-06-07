'''
    Author: Igor Martinelli - Undergraduate in Computer Science.

    This code is part of my cientific initiation in Institute of Mathematics and Computer Science (ICMC) - University of Sao Paulo.
    This code is responsible for classification
'''
def classifier_model(X_test, X_train, y_train, classifier_method) :
    if classifier_method == "KNN" :
        #RODANDO O ALGORITMO
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        classifier.fit(X_train, y_train)
    elif classifier_method == "LogisticRegression" :
        #Fitting Logistic Regression to the Training Set
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train, y_train)
        
    return classifier.predict(X_test)
            
