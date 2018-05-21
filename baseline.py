import pandas as pd
from pandas import DataFrame
import numpy as np
import csv

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk.tokenize.moses import MosesDetokenizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import metrics

#--------- edit these lines only ---------------------------------------------

PREPROCESSING     = False
SAVE_CSV          = True
FEATURE_SELECTION = False
CROSS_VALIDATION  = False

# chose vectorizer here
# to include bigrams: ngram_range=(1,2), only bigrams ngram_range=(2,2)
# remove words that occor in too many documents and too few documents set
# max_df = 0.8, min_df = 5 (more than 80% of the documents, and in less than 5 documents)

#vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b', max_df = 0.8, min_df = 5)
vectorizer = TfidfVectorizer(ngram_range=(1,2))

# chose classifier
clf = LinearSVC()
#clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)


# read training data and test data
# ----------------------------------------------------------------------------
trainData = pd.read_csv("train.tsv", header=0, delimiter='\t', quoting=3)
testData  = pd.read_csv("test.tsv", header=0, delimiter='\t', quoting=3)

# preprocessing on trainData and testData
# ----------------------------------------------------------------------------------------

if PREPROCESSING == True:    
    # helper functions and constants
    # ---------------------------------------------------------------------------------------
    defined_stop_words = ['movie','film','story', 'character', 'one', 'make', 'like', 'feel',
                          'character', 'work', 'just', 'life', 'look', 'little', 'make',
                          'director', 'come', 'year', 'watch', 'action', 'thing', 'audience',
                          'plot', 'act', 'world', 'screen',
                          'much', 'rrb', 'lrb', 'way', 'seem', 'even', 
                          '10', '100', '101', '102', '103', '104', '105',
                          '10th', '11', '110', '112', '12', '120', '127', '129', '12th',
                          '13', '13th', '14', '140', '146', '15', '15th', '16', '163',
                          '168', '170', '1790', '18', '1899', '19', '1915', '1920',
                          '1930s', '1933', '1937', '1938', '1940s', '1950', '1950s',
                          '1952', '1953', '1957', '1958', '1959', '1960', '1960s', '1962',
                          '1970', '1970s', '1971', '1972', '1973', '1975', '1979', '1980',
                          '1980s', '1984', '1986', '1987', '1989', '1990', '1991', '1992',
                          '1993', '1994', '1995', '1997', '1998', '1999', '19th', '20', '2000',
                          '2001', '2002', '20th', '21', '21st', '22', '24', '2455', '25',
                          '26', '270', '295', '30', '300', '3000', '30s', '37', '3d',
                          '40', '40s', '42', '451', '48', '4ever', '4th', '4w', '50', '500',
                          '50s', '51', '51st', '52', '53', '5ths', '60', '60s', '65', '65th',
                          '66', '70', '70s', '71', '72', '75', '77', '78', '7th',
                          '80', '800', '80s', '83', '84', '85', '86', '87', '88', '89',
                          '8th', '90', '90s', '91', '93', '94', '95', '96', '98', '99']

    # list stopwords from nltk which shall not be stopwords
    deleteStopwordFromList = set(('not', 'but'))

    # english stopwords + defined stopwords
    stop_words = text.ENGLISH_STOP_WORDS.union(defined_stop_words)
    # stopwords - stopwords that shall be normal words
    stop_words = stop_words - deleteStopwordFromList

    # lower case
    trainData["Phrase"] = trainData["Phrase"].str.lower()
    testData["Phrase"] = testData["Phrase"].str.lower()

    # remove puncutation
    trainData["Phrase"] = trainData["Phrase"].str.replace('[^\w\s]','')
    testData["Phrase"] = testData["Phrase"].str.replace('[^\w\s]','')

    # tokenization
    trainData["Phrase"] = trainData["Phrase"].apply(nltk.word_tokenize)
    testData["Phrase"] = testData["Phrase"].apply(nltk.word_tokenize)

    # remove stopwords
    trainData["Phrase"] = trainData["Phrase"].apply(lambda x: [item for item in x if item not in stop_words])
    testData["Phrase"] = testData["Phrase"].apply(lambda x: [item for item in x if item not in stop_words])

    # stemming
    stemmer = SnowballStemmer("english")
    trainData["Phrase"] = trainData["Phrase"].apply(lambda x: [stemmer.stem(y) for y in x])
    testData["Phrase"] = testData["Phrase"].apply(lambda x: [stemmer.stem(y) for y in x])

    # remove most frequent words and less frequent words
    # optional in vectorizer: done vectorizing later

    # detokenize to make it compatible with vectorizer
    detokenizer = MosesDetokenizer()
    trainData["Phrase"] = trainData["Phrase"].apply(lambda x: detokenizer.detokenize(x, return_str=True))
    testData["Phrase"] = testData["Phrase"].apply(lambda x: detokenizer.detokenize(x, return_str=True))


# vectorize the features
#------------------------------------------------------------------------------------------------------
X_train = vectorizer.fit_transform(trainData["Phrase"]) 
X_test  = vectorizer.transform(testData["Phrase"])

# print("some feature examples",vectorizer.get_feature_names()[200:250])
print("num features",len(vectorizer.get_feature_names()))


# select features
# -----------------------------------------------------------------------------------------------------
#if FEATURE_SELECTION == True:
#    
#    # add feature selector
#    selBest   = SelectKBest(chi2, k=20) # set k = "all" to run without feature selector
#    X_train   = selBest.fit_transform(X_train, trainData["Sentiment"])
#    X_test    = selBest.transform(X_test)
#    print("Automatic feature selector with k = ",k,"chosen features as best features:")
#    print(np.asarray(vectorizer.get_feature_names())[selBest.get_support()])


# training of classifier
# -----------------------------------------------------------------------------------------------------


clf.fit(X_train, trainData["Sentiment"])


# predict
#-----------------------------------------------------------------------------------------------------

testPredictions  = clf.predict(X_test)
trainPredictions = clf.predict(X_train)

#---------- some metrics to validate the model --------------------------------------------------------
# accuracy on the trainings data
#print(metrics.classification_report(trainData["Sentiment"], trainPredictions, target_names = ["negative", "somewhat negative", "neutral", "somewhat posiive", "positive"]))
#print(metrics.confusion_matrix(trainData["Sentiment"], trainPredictions))

accuracy = clf.score(X_train, trainData["Sentiment"])
print("accuracy on training data:", accuracy)


#----------- cross validation --------------------------------------------------------------------------
#if CROSS_VALIDATION == True:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train, trainData["Sentiment"], cv=5, scoring='accuracy')
print(scores) 

# save test result to csv file
#------------------------------------------------------------------------------------------------------
#if SAVE_CSV == True:
#    saveData = {'PhraseId': testData["PhraseId"], 'Sentiment': testPredictions}
#    dfResult = pd.DataFrame(data=saveData)
#
#    dfResult.to_csv("result.csv", index=False) # index=False keeps pandas from saving the index to the file











         


    



