from flask import Flask, request, render_template
import pickle
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)
dt = pickle.load(open('model.pkl', 'rb'))
LM = LM = WordNetLemmatizer()
cv = CountVectorizer()
data = pd.read_excel('Question Dataset 1.xlsx')

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
def prep(ip):
    
    cp = []
    review = re.sub('[^a-zA-Z]',' ',ip)
    review = review.lower()
    review = review.split() 
    review = [LM.lemmatize(word) for word in review if not word in stopwords.words('english') ]
    review = ''.join(review)
    cp.append(review)
    ip = cv.transform(cp)
    return ip

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = str(request.form.values())
    pred = dt.predict(prep(int_features))
    if(pred == 1):
        result = 'Knowledge'
    if(pred == 2 ):
        result = 'Understanding'
    if(pred == 3):
        result = 'Analysis'
    if(pred == 4):
        result = 'Application'   


    #response = 
    return render_template('index.html', prediction_text='The question type is {}'.format(result))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    
   '''
   For direct API calls trought request
   '''
   print(request)
   ip_features = str(request.form['question'])
   pred = dt.predict(prep(ip_features))
   if(pred == 1):
        result = 'Knowledge'
   if(pred == 2 ):
        result = 'Understanding'
   if(pred == 3):
        result = 'Analysis'
   if(pred == 4):
        result = 'Application'
   output = result
   return {'label':output}

if __name__ == "__main__":
    app.run(debug=True)