import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

class PreProcessing():
    @staticmethod
    def preprocess_csv(file):
        df = pd.read_csv(file, encoding="utf-8")
        df["cleaned_desc"] = df["description"].apply(self.cleanText)
        df["cleaned_title"] = df["title"].apply(self.cleanText)
        df["full_cleaned"] = df["cleaned_desc"] + df["cleaned_title"]
        df = df.reset_index()

        return df["full_cleaned"].values

    @staticmethod
    def cleanText(x):
        soup = BeautifulSoup(x, 'lxml')
        no_html_text = soup.get_text()
        tokens = nltk.word_tokenize(no_html_text)
        tokens = [w.lower() for w in tokens]
        words = [word for word in tokens if word.isalpha()]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        stopwords = nltk.corpus.stopwords.words('italian')
        stopwords.extend(string.punctuation)
        stopwords.extend(nltk.corpus.stopwords.words('english'))
        words = [w for w in stripped if w.isalpha() and not w in stopwords]
    
        return " ".join(words)  