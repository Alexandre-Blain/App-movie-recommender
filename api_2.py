from annoy import AnnoyIndex
from flask import Flask, jsonify, request
import os
import pandas as pd
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from nltk.corpus import stopwords
import re
import pickle
import numpy as np
import json
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import wget 
import zipfile
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from annoy import AnnoyIndex
import os
import torch
from torchtext.data.utils import get_tokenizer


vectorizer = pd.read_pickle("Fichiers_pkl/tfidf.pkl")
bag_count= pd.read_pickle("Fichiers_pkl/BoW_Count.pkl")

def bag_tidf(texte):
    
    
    matrix = vectorizer.transform([texte])

    return matrix[0]

def bag(texte):
    matrix = bag_count.transform([texte])

    return matrix[0]

def tokenize(text):
    tokenizer = get_tokenizer("basic_english")
    return [token.lower() for token in tokenizer(text)]

def description_to_embedding(description, vocab, weights_matrix):
    indices = torch.tensor([vocab[token] for token in tokenize(description)], dtype=torch.long)
    embeddings = weights_matrix[indices]
    return embeddings.mean(dim=0)



with open('Fichiers_pkl/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
    
with open('Fichiers_pkl/weights_matrix.pkl', 'rb') as f:
    weights_matrix = pickle.load(f)
    
    


app = Flask(__name__)

@app.route('/Glove', methods=['POST'])
def Glove():
    data = request.json
    df = data.get('vec')
    k = data.get('nb_reco')
    m = data.get('method')
    
    
    t = AnnoyIndex(100, 'angular')
    t.load('Fichiers_pkl/glove.ann')
    
    new_movie_embedding = description_to_embedding(df, vocab, weights_matrix)
    indices= t.get_nns_by_vector(new_movie_embedding,k)


   
    return jsonify({"prediction": indices})



@app.route('/Bag_of_words', methods=['POST'])
def Bag_of_words():
    data = request.json
    df = data.get('vec')
    k = data.get('nb_reco')
    m = data.get('method')
    
    emb_sp=bag(df)
    emb=emb_sp.toarray().flatten()
    dim=5000
    annoy_index = AnnoyIndex(dim, 'angular')
    annoy_index.load('Fichiers_pkl/BoW_Count.ann')
    
    indices = annoy_index.get_nns_by_vector(emb, k)
    return jsonify({"prediction": indices})

@app.route('/Bow_tfidf', methods=['POST'])
def Bow_tfidf():
    data = request.json
    df = data.get('vec')
    k = data.get('nb_reco')
    m = data.get('method')
    
    emb_sp=bag_tidf(df)
    emb=emb_sp.toarray().flatten()

    dim=5000
    annoy_index = AnnoyIndex(dim, 'angular')
    annoy_index.load('Fichiers_pkl/Tfidf.ann')
    indices = annoy_index.get_nns_by_vector(emb, k)
    return jsonify({"prediction": indices})

@app.route('/predict', methods=['POST'])
def predict():
 data = request.json
 df = data.get('vec')
 k = data.get('nb_reco')
 m = data.get('method')
 
 if m =="image":
     emb=df
     dim = 576
     annoy_index = AnnoyIndex(dim, 'angular')
     annoy_index.load('Fichiers_pkl/rec_imdb.ann')



 indices = annoy_index.get_nns_by_vector(emb, k)
 return jsonify({"prediction": indices})


if __name__ == "__main__":
 app.run(host='0.0.0.0', port=5000, debug=False)