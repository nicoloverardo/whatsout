# This python module was made as part of
# the Text Mining and Sentiment Analysis 
# exam project (2020).
#
# Author: Nicolò Verardo
#
# License: MIT License

import numpy as np

import multiprocessing
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

    Parameters
    ----------
    labels : {array-like or list} of shape (n_labels,)
        The name of the categories

    threshold : float (defalut=0.5)
        The threshold value used to filter categories.

    n_jobs : int (default=-1)
        The number of jobs to run in parallel. Fit, partial_fit 
        and predict are parallelized. -1 means using all processors.
    """

    def __init__(self, labels = None, threshold = 0.5,
                 n_jobs = None):

        # TODO: implement the threshold value selection
        self.threshold = threshold
        self.C = labels
        self.tfidf = OnlineTfidfVectorizer()

        if n_jobs is None or n_jobs == 0:
            self.n_jobs = 1
        elif n_jobs != -1 and n_jobs <= multiprocessing.cpu_count():
            self.n_jobs = n_jobs
        else:
            self.n_jobs = multiprocessing.cpu_count()


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

        n_categories = len(self.C)
        n_documents, n_terms = X_tfidf.shape

        self.T = self.tfidf.get_feature_names()

        # Maybe we can work with a sparse W?
        self.W = np.zeros((n_terms, n_categories))

        # TODO: parallelize
        for d in range(n_documents):
            x_nnz = X_tfidf[d,].nonzero()[1]
            y_nnz = y[d,].nonzero()[0]

            for i in x_nnz:
                for j in y_nnz:
                    self.W[i,j] += X_tfidf[d,i]


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

        X_tfidf = self.tfidf.partial_refit_transform(X)

        new_categories = set(labels) - set(self.C)
        n_new_categories = len(new_categories)
        n_new_terms = set(self.tfidf.get_feature_names()) - set(self.T)



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

            W_prime = np.dot(F, self.W)

            y[i,] = np.where(W_prime > self.threshold, 1, 0)

            #for j in range(W_prime.shape[0]):
                #y[i,j] = 1 if W_prime[j] > self.threshold else 0

        return y