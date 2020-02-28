
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

def getTFID(name,feature_file):
  if not os.path.isfile(feature_file):
    if(name=='train'):
      df = pd.read_csv('targetdir/data.csv', sep='\n', header=None)
    else:
      df = pd.read_csv('targetdir/comp_data_head_body.csv', sep='\n', header=None)

    docs = []
    for i in range(df.shape[0]):
      docs.append(' '.join(eval(df.iloc[i][0])))

    tfidf_vectorizer=TfidfVectorizer(use_idf=True,smooth_idf=True,min_df = 0.005,max_features = 500)
    tfidf_vectors=tfidf_vectorizer.fit_transform(docs)

    # pca = PCA(n_components = 100)
    # principalComponents = pca.fit_transform(tfidf_vectors.todense())
    # print("Calculated PCA...")
    np.save(feature_file, np.array(tfidf_vectors.todense()))
  return np.load(feature_file)



