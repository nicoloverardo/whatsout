# This python module was made as part of
# the Text Mining and Sentiment Analysis 
# exam project (2020).
#
# Author: Nicolò Verardo
#
# License: MIT License

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer

class MatrixRegression(BaseEstimator):
    """ 
    Implementation of the MatrixRegression (MR) algorithm
    as presented in the following paper:

    Popa, I. & Zeitouni, Karine & Gardarin, Georges & Nakache,
    Didier & Métais, Elisabeth. (2007). Text Categorization for
    Multi-label Documents and Many Categories.
    421 - 426. 10.1109/CBMS.2007.108.
    """

    def __init__(self, threshold):
        """
        Parameters
        ----------
        threshold : float
            The threshold value used to filter categories.
        """

        # TODO: set a default threshold value?
        self.threshold = threshold

        # TODO: Maybe we'll want to add the possibility to pass
        # the vectorizer as a parameter in order for the
        # user to specify its parameters manually.
        # Pass it with default value None and then assign it like:
        #
        # self.tfidf = TfidfVectorizer() if vectorizer is None else vectorizer
        #
        # where 'vectorizer' is the param name.
        # Maybe also check for the correct type...
        self.tfidf = TfidfVectorizer()        

    def fit(self, X, y, labels):
        """ 
        Fit the MatrixRegression algorithm

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

        X_tfidf = self.tfidf.fit_transform(X)

        n_categories = len(labels)
        n_terms = X_tfidf.shape[1]

        self.T = self.tfidf.get_feature_names()
        self.C = labels

        self.W = np.zeros((n_terms, n_categories))

        for d in range(n_terms):
            x_nnz = X_tfidf[d,].non_zero()[1]

            for i in x_nnz:
                y_nnz = y[i,].non_zero()[0]

                for j in y_nnz:
                    self.W[i,j] += X_tfidf[i,j]
    
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

        T = self.T
        C = self.C
        W = self.W

        y = np.zeros((len(X), len(C)))

        # Nope, don't like this.
        # TODO: can we avoid a for loop?
        for i, x in enumerate(X):
            # 'x' should be a single string of text that contains
            # the entire document. Thus, we need to split it.
            # TODO: check for the correct input handling
            # and formats.
            T_d = set(x.split())
            
            F = np.zeros(len(T))

            T_prime = T_d.intersection(T)  

            for t in T_prime:
                F[T.index(t)] = 1

            W_prime = F.dot(W)

            # Nope, don't like this too.
            # TODO: can we avoid a for loop?
            # Check also that the access to the
            # 2-dimensional array is done correctly.
            for j, c in enumerate(W_prime):
                y[i,j] = 1 if c > self.threshold else 0

        return y