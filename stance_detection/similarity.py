from gensim.models import Doc2Vec
from tqdm import tqdm
from gensim.models.doc2vec import TaggedDocument
from feature_engineering import clean, get_tokenized_lemmas
import os
import numpy as np
from sklearn import utils
import pandas as pd
from fuzzywuzzy import fuzz
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis


def find_similarity(h,b,feature_file,name):
  data_head=0
  data_body=0

  if not os.path.isfile(feature_file):
    

    if(name == 'train'):
      data_head=pd.read_csv("targetdir/bertembed/trainheadline_bert.csv").drop('Unnamed: 0',axis=1)
      data_body=pd.read_csv("targetdir/bertembed/trainbody_bert.csv").drop('Unnamed: 0',axis=1)
    elif(name == 'comp'):
      data_head=pd.read_csv("targetdir/bertembed/compheadline_bert.csv").drop('Unnamed: 0',axis=1)
      data_body=pd.read_csv("targetdir/bertembed/compbody_bert.csv").drop('Unnamed: 0',axis=1)




    X_cosine=[]
    X_city=[]
    X_jacc = []
    X_can = []
    X_euc = []
    X_min=[]
    X_bray=[]
    for i in tqdm(range(len(data_body))):
      X_cosine.append(cosine(np.array(data_head.iloc[i]),np.array(data_body.iloc[i])))
      X_city.append(cityblock(np.array(data_head.iloc[i]),np.array(data_body.iloc[i])))
      X_jacc.append(jaccard(np.array(data_head.iloc[i]),np.array(data_body.iloc[i])))
      X_can.append(canberra(np.array(data_head.iloc[i]),np.array(data_body.iloc[i])))
      X_euc.append(euclidean(np.array(data_head.iloc[i]),np.array(data_body.iloc[i])))
      X_min.append(minkowski(np.array(data_head.iloc[i]),np.array(data_body.iloc[i])))
      X_bray.append(braycurtis(np.array(data_head.iloc[i]),np.array(data_body.iloc[i])))
    

    X_qr=[]
    X_wr=[]
    X_partial=[]
    X_token=[]
    X_ptoken=[]
    for i, (headline, body) in tqdm(enumerate(zip(h, b))):
      X_qr.append(fuzz.QRatio(headline,body))
      X_wr.append(fuzz.WRatio(headline, body))
      X_partial.append(fuzz.partial_ratio(headline, body))
      X_token.append(fuzz.token_set_ratio(headline,body))
      X_ptoken.append(fuzz.partial_token_set_ratio(headline, body))

    np.save(feature_file, np.c_[X_cosine, X_city, X_jacc, X_can, X_euc, X_min, X_bray, X_qr,X_wr,X_partial,X_token,X_ptoken])
    print("Calculated distances...")
  return np.load(feature_file)
