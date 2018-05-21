# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 10:13:05 2018

@author: Fish
"""
import pandas as pd
from pandas import DataFrame
import numpy as np
import csv
import spacy
import en_core_web_sm

from sklearn.model_selection import KFold
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB

import nltk
nltk.download("sentiwordnet")
from nltk.corpus import sentiwordnet as swn

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer      

""" define protocols for syntactic parsing -----------------------------------------
"""
BADTAGS = ["PUNCT","ADP","SYM", "NUM", "DET", "PART", "PROPN"] #words with these tags will be removed
NEGATES = ["nor","not","never","no","n't"] #words preceded by these tokens will be negated
KEEP = ["none", "much"] #these stop words will NOT be removed, as they carry sentiment

#load syntactic parser
nlp = en_core_web_sm.load()

# read train data and test data
trainData = pd.read_csv("train.tsv", header=0, delimiter='\t', quoting=3)
testData = pd.read_csv("test.tsv", header=0, delimiter='\t', quoting=3)

#arrays for handling results of syntactic parse
#there are up to 60 tokens in each phrase
tokens = np.empty([len(trainData), 60], dtype="S20")
tags = np.empty([len(trainData), 60], dtype="S10")

""" syntactic pos taggin of the full training set --------------------------------
pos tag every phrase, based on syntactic parsing of the parent sentence only
(ie. pos tags are copied down to sibling phrases)
append POS tags to every word (these word|pos combination will be used to x-reference sentiment polarity from IMDB)
negate any POS tags that are preceded by a negation
then concatenate all word|pos pairs back into the pandas dataframe (as the new phrase)
"""
previous = ""
for entry, record in enumerate(trainData.values):#['Phrase'].values):
    #if entry < 10:
    #records[entry, 0] = record  
    phrase = record[1] #get current sentence index
    parsed = nlp(record[2])
    vector = ""
    if phrase == previous:
        col = 0 #position in resulting cleansed phrase
        for token in parsed: #for each token in phrase
            pos = 0  #while token doesnt match master]
            while pos < len(master) and token.lemma_ != master[pos][0]:
                pos += 1
            if pos < len(master): #if token still in master
                tokens[entry, col] = master[pos][0]
                tags[entry, col] = master[pos][1]
                vector = vector + " " + master[pos][0] + "Z" + master[pos][1] #then append token
                col += 1
        if col == 0: #if no useful tokens found, arbitrarily keep the first token
            tokens[entry, col] = parsed[0].lemma_
            tags[entry, col] = parsed[0].pos_
            vector = vector + " " + parsed[0].lemma_ + "Z" + parsed[0].pos_ #then append token
    else:
        previous = phrase
        prevtok = ""
        master = [[token.lemma_, token.pos_] for token in parsed
                  if (token.text in NEGATES + KEEP or not (token.is_stop or token.pos_ in BADTAGS))]
        col = 0 #position in resulting cleansed phrase
        cleansed = []
        if len(master) == 0: #if no useful tokens found, arbitrarily keep the first token
            cleansed.append([parsed[0].lemma_, parsed[0].pos_])
        for pos, token in enumerate(master):
            if prevtok in NEGATES:
                col -= 1
                cleansed.pop(col) #delete preceding (negator) token
                cleansed.append([token[0], "X" + token[1]])
                master[pos][1] = "X" + token[1] #must also negate master for loading into subphrases
                prevtok = ""
            else:
                cleansed.append(token)
                prevtok = token[0]
            tokens[entry, col] = token[0]
            tags[entry, col] = cleansed[col][1]
            col += 1
        vector = cleansed[0][0] + "Z" + cleansed[0][1]
        for token in cleansed[1:]:
            vector = vector + " " + token[0] + "Z" + token[1]  #then append token
            col += 1
    #print(vector)
    #trainData.loc[entry,'record'] = vector
    trainData.set_value(entry, 'Phrase', vector)

"""
export tokens, tags & features for further pre-processing --------------------------
statistical analysis of token|tags was performed offline using manual processing
"""    
#export all tokens and pos tags for further feature processing 
columns = []
for col, tok in enumerate(tokens[0]):
    columns.append("col"+str(col))
tk = pd.DataFrame(tokens, columns=columns)
tk.to_csv("tokens.csv", index=False)
tg = pd.DataFrame(tags, columns=columns)
tg.to_csv("tags.csv", index=False)

# extract features
vectorizer = CountVectorizer(lowercase=False)
X = vectorizer.fit_transform(trainData["Phrase"])#.iloc[0:10,2:3])

# save modified features to txt file, for further processing
output_file = open('features.txt', 'w')
print(vectorizer.get_feature_names(), file=output_file)#50])

# save full vectorised array to txt file, for further analysis
columns = []
for col, feature in enumerate(X[0]):
    columns.append("col"+str(col))
vc = pd.DataFrame(X, columns=columns)
vc.to_csv("vectors.csv", index=False)
# save a single sample for easier offline handling
sample = X[1,:].toarray()
sample = np.transpose(sample)
psample = pd.DataFrame(sample, columns=['col1'])
psample.to_csv("sample.csv", index=False)

"""
train and predict using the vectorised features ----------------------------------
NOTE for further tests after feature engineering
these steps were repeated for each generated feature set
"""    
# train naive bayes with extracted features
clf = MultinomialNB()
clf.fit(X, trainData["Sentiment"])
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
accuracy = clf.score(X, trainData["Sentiment"])
print(accuracy)

# repeat with 10-fold cross-validation
score = []
predictions = clf.predict(X)
k_fold = KFold(n_splits=10)
for train_indices, test_indices in k_fold.split(X):
    clf.fit(X[train_indices], X[train_indices])
    predictions[test_indices] = clf.predict(X[test_indices])
    score.append(clf.score(X[test_indices], X[test_indices]))
print('Average accuracy:', np.mean(score))

# test with naive bayes
test_counts = vectorizer.transform(testData["Phrase"])
predictions = clf.predict(test_counts)
print("some predictions (shall be between 0-4):",predictions[1:10])#100]) # [0,1,2,3,4]

# save test result to csv file
d = {'PhraseId': testData["PhraseId"], 'Sentiment': predictions}
dfResult = pd.DataFrame(data=d)

dfResult.to_csv("result.csv", index=False) # index=False keeps pandas from saving the index to the file 

"""
feature pruning ----------------------------------------------------------------
to feed these features back to the classifier, the above training code was re-used
"""
#import set of pruned (high polarity) features
with open('pruned_features.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row, feature in enumerate(readCSV):
        pruned[row] = feature
#identify columnar indexes of each prnued feature in the vectorised array    
#(pruned items have been marked with 'X')
cols = []
for col, feat in enumerate(pruned[:,0]):
    if feat != 'X':
        cols.append(col)
#then prune the vectorised array 
newX = X[:,cols].toarray()

"""
feature weighting ----------------------------------------------------------------
"""
#import set of feature weights (scalars)
test = csv.reader('feature_weights.csv', delimiter=',')

#import set of weighted features
for row, feature in enumerate(test['weights']):
    X[:,row] = X[:,row].toarray()*test.iloc[row,0] 

##-----------------------------------------
sample = X[1,:].toarray()
sample = np.transpose(sample)
psample = pd.DataFrame(sample, columns=['col1'])
psample.to_csv("sample.csv", index=False)

#
    
