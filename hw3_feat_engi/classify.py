'''
Baozhi Yu
Use one late day for this assignment
'''
from csv import DictReader, DictWriter

import numpy as np
from numpy import array

import nltk.stem
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA


kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'

token = u'dead|die|kills|killed|revenge|suicide'
english_stemmer = PorterStemmer()

class StemmedTfidfVectorizer(TfidfVectorizer) :
	def build_analyzer(self) :
		analyzer = super(StemmedTfidfVectorizer , self).build_analyzer()
		return lambda doc : (english_stemmer.stem(w) for w in analyzer(doc))

class StemmedCountVectorizer(CountVectorizer) :
	def build_analyzer(self) :
		analyzer = super(StemmedCountVectorizer , self).build_analyzer()
		return lambda doc : (english_stemmer.stem(w) for w in analyzer(doc))
		
class Featurizer:
    def __init__(self):
        #self.vectorizer = CountVectorizer(analyzer = 'char_wb' , ngram_range=(2,2) , min_df =1)
        #self.ordinary = CountVectorizer(min_df = 2 , stop_words='english')
        #self.add = CountVectorizer(token_pattern = token , min_df = 2 , stop_words='english')
        #self.vectorizer = FeatureUnion([("Bag",self.ordinary),("Add",self.add)])
        #self.vectorizer_tf = TfidfVectorizer()
        #self.vectorizer = FeatureUnion([("vec",self.vectorizer_count),("tf",self.vectorizer_tf)])
        #self.vectorizer = TfidfVectorizer(binary = True , min_df = 2)
        self.vectorizer = StemmedTfidfVectorizer(binary = True , min_df = 3 , strip_accents='unicode' ,
                          use_idf = 1 , sublinear_tf = 1 , norm = 'l2')
        
        
    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    feat = Featurizer()
    #print feat.vectorizer
    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    print("Label set: %s" % str(labels))

    x_train = feat.train_feature([x[kTEXT_FIELD] for x in train])
    x_test = feat.test_feature([x[kTEXT_FIELD] for x in test])
    #print len(feat.vectorizer.get_feature_names())
    #print x_train.shape
    y_train = array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))
    
    print(len(train), len(y_train))
    print(set(y_train))
    
    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    feat.show_top10(lr, labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["Id", "spoiler"])
    o.writeheader()
    for ii, pp in zip([x['Id'] for x in test], predictions):
        d = {'Id': ii, 'spoiler': labels[pp]}
        o.writerow(d)
