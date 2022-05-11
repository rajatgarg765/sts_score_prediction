

from flask import Flask, render_template, request
import jsonify
import requests
import os
import numpy as np
app = Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

from difflib import SequenceMatcher
def similar(a,b):
    return SequenceMatcher(None,a,b).ratio()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances

@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        str1 =str(request.form['1_para'])
        str2=str(request.form['2_para'])
        print(str1)
        print(str2)
        if len(str1)<35 or len(str2)<35:
            prediction=(similar(str1,str2))
        else:
            y=[str1,str2]
            vect=CountVectorizer()
            features1=vect.fit_transform(y).todense()
            a=[]
            for f in features1:
                a.append(euclidean_distances(features1[1],f)[0])
            prediction=(100-a[0])/100
        output=np.round(prediction,2)
        if output<0 or output>10:
            return render_template('index.html',prediction_text="PLEASE ENTER  CORRECT DETAILS")
        else:
            return render_template('index.html',prediction_text="Similarity score is  {} ".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)