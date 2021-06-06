import pickle
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
#Importing Dataset
data = pd.read_excel('Question Dataset 1.xlsx')
#Preprocessing 
data['Category']=data['Category'].replace({'Knowledge':1,'Knowlrdge':1,'Understand':2,'Understanding':2,'Understading':2,'Understanding':2,'Understanding':2,'Analysis':3,
            'Application':4,'Applicationl':4,'Understanding ':2,'Understanding\xa0':2})
data = data.rename(columns = {'Questiion ':'Ques'}, inplace = False)
LM = WordNetLemmatizer()

corpus = []
for i in range(0,len(data)):
    review = re.sub('[^a-zA-Z]',' ',data['Ques'][i])
    review = review.lower()
    review = review.split() 
    review = [LM.lemmatize(word) for word in review if not word in stopwords.words('english') ]
    review = ''.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus)

y = data['Category']

dt = DecisionTreeClassifier()
dt.fit(X, y)

filename = 'model.pkl'
pickle.dump(dt, open(filename, 'wb'))