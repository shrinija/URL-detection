from flask import Flask, render_template, request

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']

        data = pd.read_csv('spam_ham_dataset.csv')

        # Splitting the dataset into training and testing sets (80-20 split)
        X = data['text']
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        vectorizer = CountVectorizer()
        X_train_vectorized = vectorizer.fit_transform(X_train)
        
        model = MultinomialNB()
        model.fit(X_train_vectorized, y_train)

        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        result = "Not Spam" if prediction == "ham" else "Spam"
        
        return render_template('index.html', result=result, text=text)

if __name__ == '__main__':
    app.run(debug=True)
