import csv
import os
import pandas as pd
import numpy as np
import spacy
import time
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

start = time.time()
# 1st part

file = open("../data/updated_train.tsv","r")
reader = csv.reader(file,delimiter = '\t')

corpus = []
y = []

iters = iter(reader)
next(iters)
for row in iters:
	corpus.append(str(row[1]))
	y.append(int(row[2]))

vectorizer = TfidfVectorizer(max_df = 0.8, min_df= 5)
X_train = vectorizer.fit_transform(corpus)

clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y)

file = open("../data/updated_test.tsv","r")
reader = csv.reader(file,delimiter='\t')

corpus_test = []
test_tweets_ids = []

iters = iter(reader)
next(iters)
for row in iters:
	corpus_test.append(str(row[1]))
	test_tweets_ids.append(str(row[0]))

X_test = vectorizer.transform(corpus_test)

result = list(clf.predict(X_test))

i = 0
cnt = 0
file = open("../predictions/RF.csv","w")
file.write("id,hateful\n")

for i in range(0,len(result)):
	file.write(test_tweets_ids[i]+","+str(result[i])+'\n')
	if(result[i]==1):
		cnt+=1

print('number of hateful tweets(RF): ',cnt)

# 2nd part

nlp = spacy.load("en_core_web_md")

file = open("../data/updated_train.tsv","r")
reader = csv.reader(file,delimiter = '\t')
X_train_svm = []
iters = iter(reader)
next(iters)
y_svm = []

for row in iters:
	sentence = str(row[1])
	tokens = nlp(sentence)
	y_svm.append(int(row[2]))
	X_train_svm.append(tokens.vector)

X_train_svm = np.array(X_train_svm)
y_svm = np.array(y_svm)

clf = SVC()
clf.fit(X_train_svm, y_svm)

file = open("../data/updated_test.tsv","r")
reader = csv.reader(file,delimiter = '\t')
X_test = []
iters = iter(reader)
next(iters)

for row in iters:
	sentence = str(row[1])
	tokens = nlp(sentence)
	X_test.append(tokens.vector)
X_test = np.array(X_test)

result = list(clf.predict(X_test))

i = 0
cnt = 0
file = open("../predictions/SVM.csv","w")
file.write("id,hateful\n")

for i in range(0,len(result)):
	file.write(test_tweets_ids[i]+","+str(result[i])+'\n')
	if(result[i]==1):
		cnt+=1

print('number of hateful tweets(SVM): ',cnt)

#3rd part

ft_train_file = open('ft_train.txt','w')
train_file = open('../data/updated_train.tsv','r')
test_file = open('../data/updated_test.tsv','r')
reader = csv.reader(train_file,delimiter='\t')

iters = iter(reader)
next(iters)
for row in iters:
	ft_train_file.write(row[1])
	ft_train_file.write(' __label__')
	ft_train_file.write(row[2]+'\n')

model = fasttext.train_supervised('ft_train.txt',epoch=100)
os.remove('ft_train.txt')

result_file = open('../predictions/FT.csv','w')
result_file.write('id,hateful\n')
reader = csv.reader(test_file,delimiter='\t')
iters = iter(reader)
test_tweets_ids = []
next(iters)
X_test_ft = []

for row in iters:
	test_tweets_ids.append(row[0])
	X_test_ft.append(str(row[1]))


ft_result = model.predict(X_test_ft)
print(len(ft_result))
cnt = 0
print((ft_result[0][0][0]))
for i in range(0,len(ft_result[0])):
	result_bit = 1
	if(ft_result[0][i][0]=='__label__0'):
		result_bit = 0
	result_file.write(str(test_tweets_ids[i])+','+str(result_bit)+'\n')
	if(result_bit==1):
		cnt+=1

print('Number of hateful tweets(FT):', cnt)

print('Time taken: ',time.time()-start)