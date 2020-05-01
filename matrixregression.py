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
        self.vectorizer = OnlineTfidfVectorizer()
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


        X_tfidf = self.vectorizer.fit_transform(X)
        n_documents, n_terms = X_tfidf.shape

        n_categories = self._get_number_catgories(y)

        self.terms = np.array(self.vectorizer.get_feature_names(), dtype = 'object')

        self.W = np.zeros((n_terms, n_categories))

        # TODO: parallelize
        for d in range(n_documents):
            x_nnz = X_tfidf[d,].nonzero()[1]
            y_nnz = y[d,].nonzero()[0]

            self._set_weights_values(X_tfidf, x_nnz, y_nnz, d)

        return self
    
    def partial_fit(self, X, y):
        """ 
        Update the algorithm with new data without
        re-training it from scratch.

        Parameters
        ----------
        X : array-like of shape (n_documents,)
            The documents of the training collection

        y : array-like of shape (n_documents, n_labels)
            The target labels of the documents (i.e.: the categories)
        
        Returns
        -------
        self : object
        """

        old_vocab = self.vectorizer.vocabulary_.copy()

        # Get terms of the new data
        v_tmp = OnlineTfidfVectorizer()
        ct = set(v_tmp.fit(X).vocabulary_)

        # Get index of the common terms.
        # Probably this is gonna be slow
        vocab_to_update = { k : old_vocab[k] for k in ct.intersection(set(old_vocab)) }

        X_tfidf = self.vectorizer.partial_refit_transform(X)

        # Get index of the out-of-vocabulary terms only
        # Same perfomance as above?
        oov_terms = { k : self.vectorizer.vocabulary_[k] for k in set(self.vectorizer.vocabulary_) - set(old_vocab) }

        # Get all the terms to update 
        vocab_to_update.update(oov_terms)

        n_terms_to_update = len(vocab_to_update)
        n_oov_terms = len(oov_terms)

        # y must contain both the old categories
        # and the new one(s).
        n_new_categories = self._get_number_catgories(y) - self.W.shape[1]

        # Expand W.
        # Probably self.W.resize is faster?
        if n_oov_terms > 0:
            self.W = np.concatenate((self.W, 
                                     np.zeros((n_oov_terms, self.W.shape[1]))))
        if n_new_categories > 0:
            self.W = np.concatenate((self.W, 
                                     np.zeros((self.W.shape[0], n_new_categories))), axis = 1)

        terms_to_update = np.fromiter(vocab_to_update.values(), dtype=int)

        n_documents = X_tfidf.shape[0]

        self.terms = np.array(self.vectorizer.get_feature_names(), dtype = 'object')

        for d in range(n_documents):
            # Get only the terms that we need to update
            x_nnz = np.intersect1d(X_tfidf[d,].nonzero()[1], terms_to_update)

            # Still need to check that
            # x_nnz and/or y_nnz are not empty
            y_nnz = y[d,].nonzero()[0]
        
            # Set the weights of terms we need to update
            # to zero. This should be faster:
            #
            #   self.W[x_nnz, y_nnz] = 0
            #
            # Still need to test it though.

            for i in x_nnz:
                for j in y_nnz:
                    self.W[i,j] = 0

            self._set_weights_values(X_tfidf, x_nnz, y_nnz, d)

        return self

    def _set_weights_values(self, X, x_nnz, y_nnz, d):
        for i in x_nnz:
            for j in y_nnz:
                self.W[i,j] += X[d,i]

    def _get_number_catgories(self, y):
        """
        Get the number of categories from the labels.

        Parameters
        ----------
        y : array-like of shape (n_documents, n_labels)
            The target labels of the documents (i.e.: the categories)

        Returns
        -------
        n : int
            The number of categories
        """

        if isinstance(y, np.ndarray):
            if y.ndim == 2:
                return y.shape[1]
            else:
                return 1
        elif isinstance(y, list):
            return len(y)
        else:
            raise ValueError('Cannot get the number of categories.')
    
    def _compute_weights(self, X):
        """
        Compute the categories weights for new data X.

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

        tokenizer = self.vectorizer.build_tokenizer()
        y = np.zeros((X.shape[0], self.W.shape[1]), dtype=int)

        # TODO: parallelize
        for i in range(X.shape[0]):          
            T_d = np.sort(np.array(tokenizer(X[i]), dtype = 'object'))

            T_prime, x_ind, _ = np.intersect1d(self.terms, T_d, return_indices=True)

            F = np.zeros(self.terms.shape[0])
            F[x_ind] = 1

            W_prime = np.dot(F, self.W)

            y[i,] = W_prime
        
        return y

    def _predict_categories(self, y):
        """
        Filter categories using the threshold value.

        Parameters
        ----------
        y : array-like of shape (n_documents, n_labels)
            The predicted categories weights (i.e.: W')
        
        Returns
        -------
        y : array-like of shape (n_documents, n_labels)
            The predicted categories.
        """

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

        return self._predict_categories(self._compute_weights(X))