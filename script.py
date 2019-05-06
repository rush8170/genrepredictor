from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
app = Flask(__name__)

def result(to_predict):
    corpus = []
    dataset = pd.read_csv('dataset.tsv',delimiter='\t',quoting = 3, names = ['id','summary','name','genre'],nrows=1001)
    #dataset.to_csv('dataset1.tsv',sep='\t',encoding='utf-8')
    const=10
    y_corpus = []
    #print(dataset)
    for i in range(const):
        review = re.sub('[^a-zA-Z]', ' ', dataset['summary'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
        y_review = dataset['genre'][i]
        y_review = re.sub('[^0-9]', ' ', y_review)
        y_review = y_review.split()
        y_review = ','.join(y_review)
        y_corpus.append(y_review)    
    corpus.append(to_predict)
    #print(corpus)
    y = []
    for i in y_corpus:
        li = [0 for _ in range(6)]
        for j in i:
            if(j==','):
                continue
            else:
                #print(j,type(j))
                li[int(j)] = 1
        #print(li)
        y.append(li)
    
    # Creating the Bag of Words model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 20000)
    X = cv.fit_transform(corpus).toarray()
    
    X_train = X[:-1,:]
    X_test = X[-1,:]
    X_test = np.array(X_test).reshape(1,-1)
    y_train = y
    #Classifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    clfier =  OneVsRestClassifier(SVC(kernel='linear',probability=True))
    clfier.fit(X_train, np.array(y_train))
    pred = clfier.predict(X_test)
    prob = clfier.predict_proba(X_test)
    print("Probability",prob,pred)
    return prob,pred

@app.route('/result',methods=['POST'])
def res():
    if(request.method=='POST'):
        print("post request")
        plot1 = request.form
        plot = plot1['plot']
        print(plot1['plot'])
        plot = plot1['plot']
        review = re.sub('[^a-zA-Z]', ' ', plot)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        #print(review)
        prob,pred = result(review)
        #print(X)
        z = []
        if(pred[0][0]==1):
            z.append("Action")
        if(pred[0][1]==1):
            z.append("Comedy")
        if(pred[0][2]==1):
            z.append("Drama")
        if(pred[0][3]==1):
            z.append("Horror")
        if(pred[0][4]==1):
            z.append("Romance")
        if(pred[0][5]==1):
            z.append("Thriller")
    return render_template('result.html',ans = prob,res = z)

@app.route('/')
def hello_world():
    #return 'Hello, World!'
    return render_template('page.html')

if __name__ == '__main__':
    app.run(debug=True)
