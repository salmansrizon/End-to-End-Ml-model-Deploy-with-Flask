import os 
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
from visions import URL

# creating instance of the class

app =  Flask(__name__)

# index URL
@app.route('/')
@app.route('/index')

def index ():
    return render_template('index.html')

# predict function

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,12)
    loaded_model = pickle.load(open("dt_clf_gini.pickle","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
            prediction = 'Income is more then 50k'
        else:
            prediction = 'Income is less then 50k'
        return render_template('result.html', prediction = prediction)


if __name__ == '__main__':
    app.run(debug = True)