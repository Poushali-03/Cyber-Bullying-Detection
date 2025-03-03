from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

try:
    with open("stopwords.txt", "r") as file:
        stopwords = file.read().splitlines()
except FileNotFoundError:
    print("Error: stopwords.txt not found. Please check the file path.")
    stopwords = []  # Use an empty list to avoid further errors

    stopwords = stopwords if stopwords else "english"


vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True, 
                             vocabulary=pickle.load(open("tfidfvectorizer.pkl", "rb")))


# Load the trained vectorizer and model
vectorizer = pickle.load(open("tfidfvectorizer.pkl", "rb"))
model = pickle.load(open("LinearSVCTuned.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_input = request.form['headline'] 
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)