import pandas as pd 
import numpy as np
from tqdm import tqdm 
from feature_engineering import clean, get_tokenized_lemmas
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(ngram_range=(1, 10))
def terms(h,b):
    X=[]
    for i, (headline, body) in tqdm(enumerate(zip(h, b))):
            c_headline = clean(headline)
            c_body = clean(body)
            clean_headline=(get_tokenized_lemmas(c_headline))
            clean_body=(get_tokenized_lemmas(c_body))
            temphead=vect.fit_transform([' '.join(clean_headline)])
            tempbody=vect.fit_transform([' '.join(clean_body[0:40])])
            a = temphead.toarray()[0]
            b = tempbody.toarray()[0]
            print("Dot prod")
            print(np.dot(a.reshape(1,len(a)), b.reshape(1, len(b))))
            break
            X.append(np.dot(a.reshape(len(a),1),b.reshape(1,len(b))))
    print('TF-IDF done...')
    return X
