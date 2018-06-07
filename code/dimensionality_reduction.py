'''
    Author: Igor Martinelli - Undergraduate in Computer Science.

    This code is part of my cientific initiation in Institute of Mathematics and Computer Science (ICMC) - University of Sao Paulo.
    This code is responsible for dimensionality reduction of dataset
'''
from skfeature.function.similarity_based import reliefF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

def dimensionality_reduction(X_train, X_test, y_train, n_features, method):
    if method == "ReliefF":
        #produção do vetor de scores que será utilizado para seleção dos atributos.
        score = reliefF.reliefF(X_train, y_train)
        #indice dos atributos de acordo com o ranking feito pelo score.
        index = reliefF.feature_ranking(score)
        #atribuição das n_features que agora serão utilizadas em X_train e X_test             
        X_train = X_train[:, index[0:n_features]]
        X_test = X_test[:, index[0:n_features]]  
            
    elif method == "LDA":
        # Applying LDA
        lda = LDA(n_components = n_features)
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)
        
    elif method == "PCA":
        # Applying PCA
        pca = PCA(n_components = n_features)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        #explained_variance = pca.explained_variance_ratio_
    elif method == "KernelPCA":
        # Applying Kernel PCA
        kpca = KernelPCA(n_components = n_features, kernel = 'rbf')
        X_train = kpca.fit_transform(X_train)
        X_test = kpca.transform(X_test)
    
    return (X_train, X_test)
    