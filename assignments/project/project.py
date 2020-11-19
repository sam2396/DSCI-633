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
        b = resample(a, n_samples=1000, random_state=0)
        c = X.iloc[b]
        y1 = y.iloc[b]
        d = X.loc[y==1].index.values
        e = X.iloc[d]
        y2 = y.iloc[d]
        final = pd.concat([c,e])
        yf = pd.concat([y1,y2])
        final['description'] = final['description']+''+final['title']
        final = final.drop(["title","location"],axis=1)
        final["description"] = final["description"].apply(preprocess)
        final["requirements"] = final["requirements"].apply(preprocess)
        self.tfvec1 = TfidfVectorizer(stop_words='english',use_idf=True, smooth_idf=True, max_df=0.3, sublinear_tf=True,norm='l2')
        self.tfvec2 = TfidfVectorizer(stop_words='english',use_idf=True, smooth_idf=True, max_df=0.3, sublinear_tf=True,norm='l2')
        #self.preprocessor.fit(X['description'])
        self.tfvec1.fit(final["description"])
        self.tfvec2.fit(final["requirements"])
        x_trainvec_1 = self.tfvec1.transform(final["description"])
        x_trainvec_2 = self.tfvec2.transform(final['requirements'])
        final_x = pd.concat([pd.DataFrame(x_trainvec_1.todense()),pd.DataFrame(x_trainvec_2.todense())],axis=1)
        self.clf = RandomForestClassifier(n_jobs=5,n_estimators=100,criterion="entropy")
        self.clf.fit(final_x, yf)
        return

    def predict(self, X):
        X['description'] = X['description']+''+X['title']
        X1 = X.drop(["title","location"],axis=1)
        final = pd.DataFrame()
        final["description"] = X["description"].apply(preprocess)
        final["requirements"] = X["requirements"].apply(preprocess)
        x_testvec_1 = self.tfvec1.transform(final["description"])
        x_testvec_2 = self.tfvec2.transform(final['requirements'])
        final_x = pd.concat([pd.DataFrame(x_testvec_1.todense()),pd.DataFrame(x_testvec_2.todense())],axis=1)
        final_x = final_x.fillna("")
        predictions = self.clf.predict(final_x)#forloop it to get good answer
        return predictions