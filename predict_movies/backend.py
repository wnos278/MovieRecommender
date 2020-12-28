from flask import Flask, request, render_template
from pandas import DataFrame, read_csv
import pickle
# import matplotlib.pyplot as plt
# import matplotlib
import pandas as pd
import sys
kmeans = pickle.load(open("movies-prediction.pkl", "rb"))


app = Flask(__name__)
# @app.route('/')
# def my_form():
#     return render_template('client.html')

@app.route('/', methods=['POST'])


@app.route('/check', methods=['POST'])
def check():
    try:
        movie_name = request.form['movie']
        print("Movie name: ", movie_name)
    except Exception as e:
        print(e)

    ##################
    ##Predict Movies##
    ##################
    return "Recommend Movies"

if __name__ == '__main__':
    app.run()