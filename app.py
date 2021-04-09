import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from flask import Flask,render_template,request,jsonify,redirect,url_for
from Model import Model
import os
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder



classes = {
0:'Air_conditioner sound',
1:'Car horn',
2:'Children playing',
3:'Dog barking',
9:'Street music playing',
6:'Gun shot',
8:'Siren',
5:'Engine idling',
7:'Jack hammer sound',
4:'Drilling sound'
}




app = Flask(__name__)
modelInst = Model()
model = modelInst.get_model()

def feature_extractor(file):
    audio,sample_rate = librosa.load(file,res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features


@app.route('/',methods=['POST','GET'])
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        prediction_feature = feature_extractor(file_path)
        prediction_feature = prediction_feature.reshape(1,-1)
        predicted_label = model.predict_classes(prediction_feature)
        prediction_class = classes.get(predicted_label[0])
    return render_template('predict.html',data=prediction_class)






if __name__=='__main__':
    app.run(debug=True)
