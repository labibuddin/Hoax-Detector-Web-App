from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        df = pd.read_csv('scraping-dataset/valid-hoax-news.csv', encoding="latin-1")

        X = df['Judul']
        y = df['Status']

        cv = CountVectorizer()
        X = cv.fit_transform(X)  # Fit the Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Naive Bayes Classifier
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        clf.score(X_test, y_test)

        keyword = request.form['keyword']
        data = [keyword]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        return render_template('index.html', prediction=my_prediction)

    else:
        return "Unsupported Request Method"

@app.route("/index", methods=['GET'])
def index() :
    if request.method == 'GET' :
        return render_template('index.html')

@app.route("/tips", methods=['GET'])
def tips() :
    if request.method == 'GET' :
        return render_template('tips.html')

@app.route("/informasi", methods=['GET'])
def informasi() :
    if request.method == 'GET' :
        return render_template('informasi.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)