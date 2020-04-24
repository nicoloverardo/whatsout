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

    def __init__(self, threshold = 0.5, vectorizer = None):
        """
        Parameters
        ----------
        threshold : float (default=0.5)
            The threshold value used to filter categories.
        
        vectorizer : object (default=None)
            The TfidfVectorizer.
        """

        self.threshold = threshold

        # Check also that the vectorizer tokenizes
        # words and not sentences.
        if vectorizer is not None:
            if type(vectorizer) != type(TfidfVectorizer()):
                raise TypeError('The vectorizer should be of a TfidfVectorizer.')
            
            self.tfidf = vectorizer
        else:
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
        n_documents = X_tfidf.shape[0]

        self.T = self.tfidf.get_feature_names()
        self.C = labels

        self.W = np.zeros((n_terms, n_categories))

        for d in range(n_documents):
            # Get terms of the current document
            x_nnz = X_tfidf[d,].nonzero()[1]

            # Get categories of the current document
            y_nnz = y[d,].nonzero()[0]

            for i in x_nnz:
                for j in y_nnz:
                    self.W[i,j] += X_tfidf[d,i]
    
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

            for j, c in enumerate(W_prime):
                y[i,j] = 1 if c > self.threshold else 0

        return y