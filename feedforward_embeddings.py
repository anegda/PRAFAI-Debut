import re
import sys

import pickle

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from gensim.models import Word2Vec


# NECESARIOS PARA LA LIMPIEZA DE DATOS
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
    # Leemos los distintos datasets
    fTrain = sys.argv[1]+"/train.csv"
    fDev = sys.argv[1]+"/dev.csv"
    fTest = sys.argv[1]+"/test.csv"

    print("---Reading training data---")
    # -----------------train file-------------------------
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
    # 1.E- WORDEMBEDDINGS
    print("---Loading WordEmbeddings---")
    model = Word2Vec.load("./scielo_wiki_w10_c5_300_15epoch.w2vmodel")
    print("---Applying WordEmbeddings---")
    words = set(model.wv.index_to_key)
    train_df['New_Input_vect'] = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in train_df['Tokens']])
    text_vect_avg = []
    for v in train_df['New_Input_vect']:
        if v.size:
            text_vect_avg.append(v.mean(axis=0))
        else:
            text_vect_avg.append(np.zeros(300, dtype=float))
    train_df['INFORME_WORDEMBEDDINGS'] = text_vect_avg

    # TRAIN THE MODEL
    X_train = train_df[['INFORME_WORDEMBEDDINGS']]
    X_train = pd.DataFrame(X_train.INFORME_WORDEMBEDDINGS.tolist(), index=X_train.index)
    y_train = train_df[['debutFA']]
    print("---Training Model---")
    clf = MLPClassifier(random_state=1, max_iter=200).fit(X_train, y_train)
    print("---Training process has finished---")
    pickle.dump(clf, open('./baselineWE_Intersection_200.sav', 'wb'))
    print("---Model has been saved in---")

     # -----------------test file-------------------------
    print("---Reading testing data---")
    test_df = pd.read_csv(fDev, sep=",")

    # 1.- PREPROCESS
    # 1.A- Clean
    train_df["INFORME_LIMPIO"] = train_df.informe.apply(limpiar_texto)
    # 1.B- Tokenize
    train_df["Tokens"] = train_df.INFORME_LIMPIO.apply(tokenizar)
    # 1.C- Delete stopwords
    train_df["Tokens"] = train_df.Tokens.apply(eliminar_stopwords)
    # 1.D- Stem
    train_df["Tokens"] = train_df.Tokens.apply(estemizar)
    # 1.E- WORDEMBEDDINGS
    words = set(model.wv.index_to_key)
    test_df['New_Input_vect'] = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in test_df['Tokens']])
    text_vect_avg = []
    for v in test_df['New_Input_vect']:
        if v.size:
            text_vect_avg.append(v.mean(axis=0))
        else:
            text_vect_avg.append(np.zeros(300, dtype=float))
    test_df['INFORME_WORDEMBEDDINGS'] = text_vect_avg

    #EVALUATE WITH DEV
    X_test = test_df[['INFORME_WORDEMBEDDINGS']]
    X_test = pd.DataFrame(X_test.INFORME_WORDEMBEDDINGS.tolist(), index=X_test.index)
    y_test = test_df[['debutFA']]

    print("---Evaluating model---")
    print(clf.score(X_test, y_test))
    print(classification_report(y_test, clf.predict(X_test)))

def evaluate():
    fTest = sys.argv[1]+"/test.csv"

    # -----------------test file-------------------------
    print("---Reading testing data---")
    test_df = pd.read_csv(fTest, sep=",")

    print("---Loading embeddings and classifier---")
    model = Word2Vec.load("./scielo_wiki_w10_c5_300_15epoch.w2vmodel")
    clf = pickle.load(open('./baselineWE_Union_200.sav', 'rb'))

    # 1.- PREPROCESS
    # 1.A- Clean
    test_df["INFORME_LIMPIO"] = test_df.informe.apply(limpiar_texto)
    # 1.B- Tokenize
    test_df["Tokens"] = test_df.INFORME_LIMPIO.apply(tokenizar)
    # 1.C- Delete stopwords
    test_df["Tokens"] = test_df.Tokens.apply(eliminar_stopwords)
    # 1.D- Stem
    test_df["Tokens"] = test_df.Tokens.apply(estemizar)
    # 1.E- WORDEMBEDDINGS
    words = set(model.wv.index_to_key)
    test_df['New_Input_vect'] = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in test_df['Tokens']])
    text_vect_avg = []
    for v in test_df['New_Input_vect']:
        if v.size:
            text_vect_avg.append(v.mean(axis=0))
        else:
            text_vect_avg.append(np.zeros(300, dtype=float))
    test_df['INFORME_WORDEMBEDDINGS'] = text_vect_avg

    # EVALUATE WITH TEST
    X_test = test_df[['INFORME_WORDEMBEDDINGS']]
    X_test = pd.DataFrame(X_test.INFORME_WORDEMBEDDINGS.tolist(), index=X_test.index)
    y_test = test_df[['debutFA']]

    print("---Evaluating model---")
    print(clf.score(X_test, y_test))
    print(classification_report(y_test, clf.predict(X_test), digits=3))

if(sys.argv[2]=='2'):
    evaluate()
else:
    baseline()