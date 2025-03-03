from flask import Flask, render_template, request
import pickle
import gdown # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__)

# Google Drive file IDs
TFIDF_VECTOR_ID = "1GT7O_qDt74zeYOvO-QFoF1AIFio50c02"
MODEL_ID = "1RmHDZqswthpY7c3TbZMUa6JBuYTA_DvX"

# File names for local storage
TFIDF_VECTOR_FILE = "tfidfvectorizer.pkl"
MODEL_FILE = "LinearSVCTuned.pkl"

# Download function
def download_file(file_id, output):
    if not os.path.exists(output):
        print(f"Downloading {output} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
    else:
        print(f"{output} already exists. Skipping download.")

# Download model and vectorizer if not present
download_file(TFIDF_VECTOR_ID, TFIDF_VECTOR_FILE)
download_file(MODEL_ID, MODEL_FILE)

# Load the TF-IDF vectorizer and model
try:
    with open(TFIDF_VECTOR_FILE, "rb") as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("Error: TF-IDF vectorizer file not found. Please check the file path.")

try:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: Model file not found. Please check the file path.")

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
