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

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler

from online_vectorizers.online_vectorizers import OnlineTfidfVectorizer

class MatrixRegression(BaseEstimator, ClassifierMixin):
    """ 
    Implementation of the MatrixRegression (MR) algorithm
    as presented in the following paper:

    Popa, I. & Zeitouni, Karine & Gardarin, Georges & Nakache,
    Didier & Métais, Elisabeth. (2007). Text Categorization for
    Multi-label Documents and Many Categories.
    421 - 426. 10.1109/CBMS.2007.108.

    Parameters
    ----------
    threshold : float (defalut=None)
        The threshold value used to filter categories.
        Must be in the range (0, 1).

    n_jobs : int (default=None)
        The number of jobs to run in parallel. Fit, partial_fit 
        and predict will be parallelized. -1 means using all processors.
    """

    def __init__(self, threshold=None, n_jobs=None):
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.tfidf = OnlineTfidfVectorizer()
        self.scaler = MinMaxScaler(copy = False)


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

        if self.threshold is not None:
            if self.threshold <= 0 or self.threshold >= 1:
                raise ValueError('The threshold must be between 0 and 1.')

        if self.n_jobs is None or self.n_jobs == 0:
            self.n_jobs = 1
        elif self.n_jobs != -1 and self.n_jobs <= multiprocessing.cpu_count():
            self.n_jobs = self.n_jobs
        else:
            self.n_jobs = multiprocessing.cpu_count()

        if isinstance(y, np.ndarray):
            if y.ndim >= 2:
                n_categories = y.shape[1]
            else:
                n_categories = 1
        elif isinstance(y, list):
            n_categories = len(y)
        else:
            raise ValueError('Cannot get the number of categories') 


        X_tfidf = self.tfidf.fit_transform(X)
        n_documents, n_terms = X_tfidf.shape

        self.T = np.array(self.tfidf.get_feature_names(), dtype = 'object')

        # Maybe we can work with a sparse W?
        self.W = np.zeros((n_terms, n_categories))

        # TODO: parallelize
        for d in range(n_documents):
            x_nnz = X_tfidf[d,].nonzero()[1]
            y_nnz = y[d,].nonzero()[0]

            for i in x_nnz:
                for j in y_nnz:
                    self.W[i,j] += X_tfidf[d,i]

        return self
    
    def partial_fit(self, X, y, old_labels, new_labels):
        """ 
        Update the algorithm with new data without
        re-training it from scratch.

        Parameters
        ----------
        X : array-like of shape (n_documents,)
            The documents of the training collection

        y : array-like of shape (n_documents, n_labels)
            The target labels of the documents (i.e.: the categories)

        old_labels : {array-like or list} of shape (n_labels,)
            The name of the categories used to fit the algorithm.
        
        new_labels : {array-like or list} of shape (n_labels,)
            The name of the categories for the new data X.
        
        Returns
        -------
        self : object
        """


        old_vocab = self.tfidf.vocabulary_

        X_tfidf = self.tfidf.partial_refit_transform(X)

        # Probably this is gonna be slow
        new_vocab = { k : self.tfidf.vocabulary_[k] for k in set(self.tfidf.vocabulary_) - set(old_vocab) }

        n_new_terms = len(new_vocab)

        # If y is 2-dim, we can compute this like:
        #   n_new_categories = y.shape[1] - self.W.shape[1]
        # so that the labels list are not needed.
        n_new_categories = len(set(new_labels) - set(old_labels))

        # Expand W.
        # Probably self.W.resize is faster?
        if n_new_terms > 0:
            self.W = np.concatenate((self.W, np.zeros((n_new_terms,))))
        if n_new_categories > 0:
            self.W = np.concatenate((self.W, np.zeros((n_new_categories,))), axis = 1)

        new_terms = np.fromiter(new_vocab.values(), dtype=int)

        n_documents, n_terms = X_tfidf.shape

        self.T = np.array(self.tfidf.get_feature_names(), dtype = 'object')

        # TODO test this
        for d in range(n_documents):
            # Get only the terms that we need to update
            x_nnz = np.intersect1d(X_tfidf[d,].nonzero()[1], new_terms)

            y_nnz = y[d,].nonzero()[0]

            # TODO change this to a faster way
            for i in x_nnz:
                for j in y_nnz:
                    self.W[i,j] = 0

            for i in x_nnz:
                for j in y_nnz:
                    self.W[i,j] += X_tfidf[d,i]

        return self


    def _predict_weights(self, X):
        """
        Compute the categories weights for new data X

        Parameters
        ----------
        X : array-like of shape (n_documents,)
            The documents whose categories are to be
            predicted.

        Returns
        -------
        y : array-like of shape (n_documents, n_labels)
            The predicted categories weights (i.e.: W').
        """


        tokenizer = self.tfidf.build_tokenizer()
        y = np.zeros((len(X), self.W.shape[1]), dtype=int)

        # TODO: parallelize
        for i in range(X.shape[0]):          
            T_d = np.sort(np.array(tokenizer(X[i]), dtype = 'object'))

            T_prime, x_ind, _ = np.intersect1d(self.T, T_d, return_indices=True)

            F = np.zeros(self.T.shape[0])
            F[x_ind] = 1

            W_prime = np.dot(F, self.W)

            y[i,] = W_prime
        
        return y


    def _predict_categories(self, y):
        """
        Filter categories using the threshold value

        Parameters
        ----------
        y : array-like of shape (n_documents, n_labels)
            The predicted categories weights (i.e.: W')
        
        Returns
        -------
        y : array-like of shape (n_documents, n_labels)
            The predicted categories.
        """


        # Scale between 0 and 1
        y = self.scaler.fit_transform(y)

        # Use the median when the threshold is not specified
        if self.threshold is None:
            y = np.where(y > np.median(y), 1, 0)
        else:
            y = np.where(y > self.threshold, 1, 0)

        return y


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


        return self._predict_categories(self._predict_weights(X))