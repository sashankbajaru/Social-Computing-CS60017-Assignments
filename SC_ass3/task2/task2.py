import csv
import pandas as pd
import numpy as np
import spacy
import time
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

start = time.time()

nlp = spacy.load("en_core_web_md")

file = open("../data/updated_train.tsv","r")
reader = csv.reader(file,delimiter = '\t')
X_train_mlp = []
iters = iter(reader)
next(iters)
y_mlp = []

for row in iters:
	sentence = str(row[1])
	tokens = nlp(sentence)
	y_mlp.append(int(row[2]))
	X_train_mlp.append(tokens.vector)

X_train_mlp = np.array(X_train_mlp)
y_mlp = np.array(y_mlp)

#scores = cross_validate(MLPClassifier(hidden_layer_sizes=(100,50),random_state=1,max_iter=400), X_train_mlp,y_mlp, scoring=('f1_macro', 'f1_micro'), return_train_score=True)
#print('MLP: ',scores)
clf = MLPClassifier(random_state=1,max_iter=400).fit(X_train_mlp, y_mlp)

file = open("../data/updated_test.tsv","r")
reader = csv.reader(file,delimiter = '\t')
X_test = []
iters = iter(reader)
test_tweets_ids = []
next(iters)

for row in iters:
	sentence = str(row[1])
	tokens = nlp(sentence)
	test_tweets_ids.append(row[0])
	X_test.append(tokens.vector)

result = list(clf.predict(X_test))


i = 0
cnt = 0
file = open("../predictions/T2.csv","w")
file.write("id,hateful\n")

for i in range(0,len(result)):
	file.write(test_tweets_ids[i]+","+str(result[i])+'\n')
	if(result[i]==1):
		cnt+=1

print('cnt of hateful tweets: ',cnt)

print('time: ',time.time()-start)