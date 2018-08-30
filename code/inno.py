########## import required modules
import urllib2
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier


########## Function to convert html into normal text

def convertToText(html):
    soup = BeautifulSoup(html)
    relevent=soup.find_all(['p'])
    fin=""
    for r in relevent:
        fin=fin+str(r)
    return fin


########## Load train data into dataframe

data=pd.read_csv("train/train.csv")
datalen=len(data)
fixedsize=40000
categList={"news":[],
          "clinicalTrials":[],
           "publication":[],
           "guidelines":[],
           "forum":[],
           "profile":[],
           "conferences":[],
           "thesis":[],
           "others":[],
          }
for index, row in data.iterrows():
    if(row['Tag']=="news"):
        categList["news"].append(row['Webpage_id'])
    if(row['Tag']=="clinicalTrials"):
        categList["clinicalTrials"].append(row['Webpage_id'])
    if(row['Tag']=="publication"):
        categList["publication"].append(row['Webpage_id'])
    if(row['Tag']=="guidelines"):
        categList["guidelines"].append(row['Webpage_id'])
    if(row['Tag']=="forum"):
        categList["forum"].append(row['Webpage_id'])
    if(row['Tag']=="profile"):
        categList["profile"].append(row['Webpage_id'])
    if(row['Tag']=="conferences"):
        categList["conferences"].append(row['Webpage_id'])
    if(row['Tag']=="thesis"):
        categList["thesis"].append(row['Webpage_id'])
    if(row['Tag']=="others"):
        categList["others"].append(row['Webpage_id'])       
for key, value in categList.iteritems():
    totake=(len(value)*fixedsize)/datalen
    value=random.sample(value,totake)
    categList[key]=value


########## split the data into train and test data

traindata={"news":[],
          "clinicalTrials":[],
           "publication":[],
           "guidelines":[],
           "forum":[],
           "profile":[],
           "conferences":[],
           "thesis":[],
           "others":[],
          }
testdata={"news":[],
          "clinicalTrials":[],
           "publication":[],
           "guidelines":[],
           "forum":[],
           "profile":[],
           "conferences":[],
           "thesis":[],
           "others":[],
          }
for key, val in categList.iteritems():
    for index in range(len(val)):
        if(index%3==0):
            testdata[key].append(val[index])
        else:
            traindata[key].append(val[index])


traindatalist=[]
testdatalist=[]
for key, val in traindata.iteritems():
    for index in range(len(val)):
        traindatalist.append(val[index])
for key, val in testdata.iteritems():
    for index in range(len(val)):
        testdatalist.append(val[index])



########## get targets for both train and test data

targets=[]
for index,row in data.iterrows():
    if row['Webpage_id'] in traindatalist:
        targets.append(row['Tag'] )
test_targets=[]
for index,row in data.iterrows():
    if row['Webpage_id'] in testdatalist:
        test_targets.append(row['Tag'])


trainlistmin=[]
for i in range(79345):
    if i+1 not in traindatalist:
        trainlistmin.append(i+1)



testlistmin=[]
for i in range(79345):
    if i+1 not in testdatalist:
        testlistmin.append(i+1)


skip1=sorted(trainlistmin)
skip2=sorted(testlistmin)

dataList=[]
testList=[]

def pro1(mydata):
    for html in mydata['Html']:
        dataList.append(convertToText(html))


def pro2(mydata):
    for html in mydata['Html']:
        testList.append(convertToText(html))

chunksize=5000
for chunk in pd.read_csv("train/html_data.csv",chunksize=chunksize,skiprows=skip1):
    pro1(chunk)


for chunk in pd.read_csv("train/html_data.csv",chunksize=chunksize,skiprows=skip2):
    pro2(chunk)


########## get the important word count vector for the train data

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(dataList)
print X_train_counts.shape


########## calculate tfidf from the count vector calculated in the above step

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print X_train_tfidf.shape

########## Train a naive bayesian classifier

clf = MultinomialNB().fit(X_train_tfidf, targets)




text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf = text_clf.fit(dataList, targets)


########## Prediction using naive bayesian classifier

predicted = text_clf.predict(testList)
print np.mean(predicted == test_targets)


########## train a svm classfier


text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])

text_clf_svm = text_clf_svm.fit(dataList, targets)


########## prediction using svm classifier
predicted_svm = text_clf_svm.predict(testList)
print np.mean(predicted_svm == test_targets)

########## Load test data
preddata=pd.read_csv("train/test_nvPHrOx.csv")
preddoc=preddata['Webpage_id'].tolist()

predmin=[]
for i in range(79345):
    if i+1 not in preddoc:
        predmin.append(i+1)

skip=sorted(predmin)


fullpreddata=pd.read_csv("train/html_data.csv",skiprows=skip)


predList=[]

def process(mydata):
    for html in mydata['Html']:
        predList.append(convertToText(html))


for chunk in pd.read_csv("train/html_data.csv",chunksize=chunksize,skiprows=skip):
    process(chunk)

########## Prediction for the test data using svm classifier
predicted_svm_pred = text_clf_svm.predict(predList)

########## create a dataframe containing the predictons and corresponding document ids

outdf=pd.DataFrame({'Webpage_id':preddoc})
outdf['Tag']=predicted_svm_pred

########## saving the final results into a new csv file

outdf.to_csv("final_submit.csv",index=False)

