import pandas as pd
import time
import sys
sys.path.insert(0, '..')
from assignment8.my_evaluation import my_evaluation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import gensim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from gensim.parsing.porter import PorterStemmer

def preprocess(text):
    ps = PorterStemmer()
    default_stopwords=gensim.parsing.preprocessing.STOPWORDS
    split = text.split()
    for word in split :
        if word in default_stopwords :
            word = ''
        else :
            ps.stem(word)
    return ' '.join([word for word in split])


class my_model():
    def fit(self, X, y):
        a = X.loc[y==0].index.values
        b = resample(a, n_samples=1300, random_state=0)
        c = X.iloc[b]
        y1 = y.iloc[b]
        d = X.loc[y==1].index.values
        e = X.iloc[d]
        y2 = y.iloc[d]
        final = pd.concat([c,e])
        yf = pd.concat([y1,y2])
        final = final.drop(["title","location"],axis=1)
        final["description"] = final["description"].apply(preprocess)
        final["requirements"] = final["requirements"].apply(preprocess)
        self.tfvec1 = TfidfVectorizer(max_features = 2000,use_idf=False, smooth_idf=False)
        self.tfvec2 = TfidfVectorizer(max_features = 2000,use_idf=False, smooth_idf=False)
        #self.preprocessor.fit(X['description'])
        self.tfvec1.fit(final["description"])
        self.tfvec2.fit(final["requirements"])
        x_trainvec_1 = self.tfvec1.transform(final["description"])
        x_trainvec_2 = self.tfvec2.transform(final['requirements'])
        x_trainvec_1_f = pd.DataFrame(x_trainvec_1.todense())
        x_trainvec_2_f = pd.DataFrame(x_trainvec_2.todense())
        final = final.drop(["description","requirements"],axis=1)
        final_x = pd.concat([x_trainvec_1_f,x_trainvec_2_f],axis=1)
        self.clf = RandomForestClassifier(n_jobs=5,n_estimators=100,criterion="gini")
        self.clf.fit(final_x, yf)
        return

    def predict(self, X):
        X1 = X.drop(["title","location"],axis=1)
        final = pd.DataFrame()
        final["description"] = X["description"].apply(preprocess)
        final["requirements"] = X["requirements"].apply(preprocess)
        x_testvec_1 = self.tfvec1.transform(X1["description"])
        x_testvec_2 = self.tfvec2.transform(X['requirements'])
        x_testvec_1_f = pd.DataFrame(x_testvec_1.todense())
        x_testvec_2_f = pd.DataFrame(x_testvec_2.todense())
        fin = X1.drop(["description","requirements"],axis=1)
        final_x = pd.concat([x_testvec_1_f,x_testvec_2_f],axis=1)
        final_x = final_x.fillna("")
        predictions = self.clf.predict(final_x)
        return predictions




