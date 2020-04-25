# This python module was made as part of
# the Text Mining and Sentiment Analysis 
# exam project (2020).
#
# Author: Nicolò Verardo
#
# License: MIT License

import numpy as np

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer

from online_vectorizers.online_vectorizers import OnlineTfidfVectorizer

class MatrixRegression(BaseEstimator):
    """ 
    Implementation of the MatrixRegression (MR) algorithm
    as presented in the following paper:

    Popa, I. & Zeitouni, Karine & Gardarin, Georges & Nakache,
    Didier & Métais, Elisabeth. (2007). Text Categorization for
    Multi-label Documents and Many Categories.
    421 - 426. 10.1109/CBMS.2007.108.
    """

    def __init__(self, labels, threshold = 0.5):
        """
        Parameters
        ----------
        labels : {array-like or list} of shape (n_labels,)
            The name of the categories

        threshold : float (defalut=0.5)
            The threshold value used to filter categories.
        """

        # TODO: implement the threshold value selection
        self.threshold = threshold

        self.labels = labels

        self.tfidf = OnlineTfidfVectorizer()

    def fit(self, X, y):
        """ 
        Fit the MatrixRegression algorithm
        
        Parameters
        ----------
        X : array-like of shape (n_documents,)
            The documents of the training collection.

        y : array-like of shape (n_documents, n_labels)
            The target labels of the documents (i.e.: the categories)
        
        Returns
        -------
        self : object
        """

        X_tfidf = self.tfidf.fit_transform(X)

        n_categories = len(labels)
        n_terms = X_tfidf.shape[1]
        n_documents = X_tfidf.shape[0]

        self.T = self.tfidf.get_feature_names()
        self.C = labels

        # Maybe we can work with a sparse W?
        self.W = np.zeros((n_terms, n_categories))

        for d in range(n_documents):
            # Get terms of the current document
            x_nnz = X_tfidf[d,].nonzero()[1]

            # Get categories of the current document
            y_nnz = y[d,].nonzero()[0]

            for i in x_nnz:
                for j in y_nnz:
                    self.W[i,j] += X_tfidf[d,i]

        return self

    def partial_fit(self, X, y, labels):
        """ 
        Update the algorithm with new data without
        re-training it from scratch.

        Parameters
        ----------
        X : array-like of shape (n_documents,)
            The documents of the training collection

        y : array-like of shape (n_documents, n_labels)
            The target labels of the documents (i.e.: the categories)

        labels : {array-like or list} of shape (n_labels,)
            The name of the categories
        
        Returns
        -------
        self : object
        """

        raise NotImplementedError('Yet to be implemented.')

    
    def predict(self, X):
        """ 
        Predict categories for the documents in X

        Parameters
        ----------
        X : array-like of shape (n_documents,)
            The documents whose categories are to be
            predicted.

        Returns
        -------
        y : array-like of shape (n_documents, n_labels)
            The predicted categories.
        """

        y = np.zeros((len(X), len(self.C)), dtype=int)

        # TODO: parallelize
        for i in range(X.shape[0]):
            T_d = set(X[i].split())
            
            F = np.zeros(len(self.T))

            T_prime = T_d.intersection(self.T)  

            for t in T_prime:
                F[self.T.index(t)] = 1

            W_prime = F.dot(self.W)

            for j in range(W_prime.shape[0]):
                y[i,j] = 1 if W_prime[j] > self.threshold else 0

        return y

    # Do we need this in order to use MR
    # with sklearn functions? (i.e.: for cross validation)
    #def score(self, X, y):
        #raise NotImplementedError('Yet to be implemented.')