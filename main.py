import numpy as np

import sys
import pickle
import string
import scipy

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from skmultilearn.problem_transform import ClassifierChain, LabelPowerset

from text import PreProcessing

# import tensorflow as tf
# import tensorflow.keras.backend as K
# import tensorflow.keras.utils
# import tensorflow.keras.callbacks
# from tensorflow.python.client import device_lib
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten, Conv1D, GlobalMaxPooling1D
# from tensorflow.keras.callbacks import EarlyStopping

def load_file(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def make_prediction(model, vect, text):
    tokenized = PreProcessing.cleanText(text)

    if vect != None:
        X = vect.transform([tokenized])
    else:
        X = np.array([text])

    pred = model.predict(X)

    with open('models/mlb.pk', 'rb') as file:
            mlb = pickle.load(file)
            print("   Result: ", mlb.inverse_transform(pred))
    

if __name__ == '__main__':
    msg = """   Choose your prediction model carefully.

        1. [R]andom Forest
        2. [K]NN
        3. [N]eural Network
        4. [S]upport Vector Machine
        5. [M]atrix Regression

   Write [E] if you are not sure and you need time to think.\n
   Enter your wish: """

    print("""   Welcome to the genie! Remember, you have only 3 wishes.""")
    while True:
        action = input(msg).upper()

        if action == 'E':
            break
        if action not in "RKNSM" or len(action) != 1:
            action = input("\n   I don't know how to do that.\n"\
            "   Press [E] to exit or any other letter to try again: ").upper()

            if action == 'E':
                break
            else:
                continue

        text = input("""   Enter the text of the event you want to predict: """)

        if action == 'R':
            model = load_file('models/model_rf.pk')
            vect = load_file('models/bow_vectorizer.pk')

            make_prediction(model, vect, text)
            break
        elif action == 'K':
            model = load_file('models/model_knn.pk')
            vect = load_file('models/bow_vectorizer.pk')

            make_prediction(model, vect, text)
            break
        elif action == 'N':
            print('Not yet implemented')
            break
        elif action == 'S':
            model = load_file('models/model_svm.pk')
            vect = load_file('models/bow_vectorizer.pk')

            make_prediction(model, vect, text)
            break
        elif action == 'M':
            model = load_file('models/model_mr.pk')

            make_prediction(model, None, text)
            break