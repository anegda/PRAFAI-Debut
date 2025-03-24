import itertools
import re
import sys

import pandas as pd

import numpy as np

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("spanish"))
stemmer = SnowballStemmer("spanish")

def limpiar_texto(texto):
    # Remove the special characters
    texto = re.sub(r'\W', ' ', str(texto))
    # Remove words with only one character
    texto = re.sub(r'\s+[a-zA-Z]\s+', ' ', texto)
    # Replace the blanks with a single blank.
    texto = re.sub(r'\s+', ' ', texto, flags=re.I)
    # Convert text to lower case
    texto = texto.lower()
    return texto

def tokenizar(texto):
    tokens = texto.split(" ")
    return tokens

def eliminar_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]

def estemizar(tokens):
    return [stemmer.stem(token) for token in tokens]

def baseline():
    # Load datasets
    fTrain = sys.argv[1]+"/train.csv"
    fDev = sys.argv[1]+"/dev.csv"
    fTest = sys.argv[1]+"/test.csv"

    # -----------------train file-------------------------
    print("---Reading training data---")
    train_df = pd.read_csv(fTrain, sep=",")
    column_names = list(train_df.columns.values)
    print(column_names)
    # 1.- PREPROCESS
    # 1.A- Clean
    train_df["INFORME_LIMPIO"] = train_df.informe.apply(limpiar_texto)
    # 1.B- Tokenize
    train_df["Tokens"] = train_df.INFORME_LIMPIO.apply(tokenizar)
    # 1.C- Delete stopwords
    train_df["Tokens"] = train_df.Tokens.apply(eliminar_stopwords)
    # 1.D- Stem
    train_df["Tokens"] = train_df.Tokens.apply(estemizar)
    # 1.E- TF-IDF
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(train_df['INFORME_LIMPIO'])
    pickle.dump(vectorizer, open('./vectorizer_Intersection200.pk', 'wb'))
    print("---Vectorizer has been saved in---")
    train_df['INFORME_TFIDF'] = x.toarray().tolist();
    print(train_df)

    # TRAIN THE MODEL
    X_train = train_df[['INFORME_TFIDF']]
    X_train = pd.DataFrame(X_train.INFORME_TFIDF.tolist(), index=X_train.index)
    y_train = train_df[['debutFA']]
    print("---Training Model---")
    clf = MLPClassifier(random_state=1, max_iter=200).fit(X_train, y_train)
    print("---Training process has finished---")
    pickle.dump(clf, open('./baseline_Intersection200.sav', 'wb'))
    print("---Model has been saved---")

    # -----------------test file-------------------------
    print("---Reading testing data---")
    test_df = pd.read_csv(fDev, sep=",")

    # 1.- PREPROCESS
    # 1.A- Clean
    test_df["INFORME_LIMPIO"] = test_df.informe.apply(limpiar_texto)
    # 1.B- Tokenize
    test_df["Tokens"] = test_df.INFORME_LIMPIO.apply(tokenizar)
    # 1.C- Delete stopwords
    test_df["Tokens"] = test_df.Tokens.apply(eliminar_stopwords)
    # 1.D- Stem
    test_df["Tokens"] = test_df.Tokens.apply(estemizar)
    # 1.E- TF-IDF
    x = vectorizer.transform(test_df['INFORME_LIMPIO'])
    test_df['INFORME_TFIDF'] = x.toarray().tolist();
    print(test_df)

    # EVALUATE WITH DEV
    X_test = test_df[['INFORME_TFIDF']]
    X_test = pd.DataFrame(X_test.INFORME_TFIDF.tolist(), index=X_test.index)
    y_test = test_df[['debutFA']]

    print("---Evaluating model---")
    print(clf.score(X_test, y_test))
    print(classification_report(y_test, clf.predict(X_test)))

def evaluate():
    fTest = sys.argv[1]+"/test.csv"

    print("---Reading testing data---")
    test_df = pd.read_csv(fTest, sep=",")

    print("---Loading vectorizer and classifier---")
    vectorizer = pickle.load(open('./vectorizer_Union200.pk', 'rb'))
    clf = pickle.load(open('./baseline_Union200.sav', 'rb'))

    # 1.- PREPROCESS
    print("---Preproccessing---")
    # 1.A- Clean
    test_df["INFORME_LIMPIO"] = test_df.informe.apply(limpiar_texto)
    # 1.B- Tokenize
    test_df["Tokens"] = test_df.INFORME_LIMPIO.apply(tokenizar)
    # 1.C- Delete stopwords
    test_df["Tokens"] = test_df.Tokens.apply(eliminar_stopwords)
    # 1.D- Stem
    test_df["Tokens"] = test_df.Tokens.apply(estemizar)
    # 1.E- TF-IDF
    x = vectorizer.transform(test_df['INFORME_LIMPIO'])
    test_df['INFORME_TFIDF'] = x.toarray().tolist();
    print(test_df)

    #EVALUATE WITH TEST
    X_test = test_df[['INFORME_TFIDF']]
    X_test = pd.DataFrame(X_test.INFORME_TFIDF.tolist(), index=X_test.index)
    y_test = test_df[['debutFA']]

    print("---Evaluating model---")
    print(clf.score(X_test, y_test))
    print(classification_report(y_test, clf.predict(X_test), digits=3))

def sort_coo(coo_matrix):
    """Sort a dict with highest score"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature, score
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results

def get_keywords(vectorizer, feature_names, doc):
    """Return top k keywords from a doc using TF-IDF method"""

    # generate tf-idf for the given document
    tf_idf_vector = vectorizer.transform([doc])

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    # extract only TOP_K_KEYWORDS
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)

    return list(keywords.keys())

def keywords():
    print("---Loading vectorizer---")
    vectorizer = pickle.load(open('./vectorizer.pk', 'rb'))

    fTest = sys.argv[1]+"/test.csv"
    print("---Reading testing data---")
    test_df = pd.read_csv(fTest, sep=",")
    print("---Preproccessing---")
    test_df["INFORME_LIMPIO"] = test_df.informe.apply(limpiar_texto)
    test_df["Tokens"] = test_df.INFORME_LIMPIO.apply(tokenizar)
    test_df["Tokens"] = test_df.Tokens.apply(eliminar_stopwords)
    test_df["Tokens"] = test_df.Tokens.apply(estemizar)
    vectorizer.fit_transform(test_df['INFORME_LIMPIO'])

    print("---Top n terms---")
    feature_names = vectorizer.get_feature_names()
    result = []
    for doc in test_df['INFORME_LIMPIO']:
        df = {}
        df['full_text'] = doc
        df['top_keywords'] = get_keywords(vectorizer, feature_names, doc)
        result.append(df)

    pd.DataFrame(result).to_csv("palabrasRelevantes_Inter.csv")

baseline()
evaluate()
#keywords()