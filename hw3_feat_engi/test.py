from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion

train = [
    'The quick brown fox jumped over the lazy lazy dog' ,
    'There is that dog and fox again'
]

bag_of_words = CountVectorizer(stop_words = 'english')
#print bag_of_words
X = bag_of_words.fit_transform(train)
#print X
print "The named features are" , bag_of_words.get_feature_names()

print "X has type" , type(X)
print "X has shape" , X.shape

print X.todense

add_on = CountVectorizer(stop_words = 'english' , token_pattern = r'lazy')

allmyfeatures = FeatureUnion([("bag-of-words" , bag_of_words) , ("add_ons" , add_on)])

print allmyfeatures
Z = allmyfeatures.fit_transform(train)
print Z
print allmyfeatures.get_feature_names()

print "Z has type" , type(Z)
print "Z has shape" , Z.shape
