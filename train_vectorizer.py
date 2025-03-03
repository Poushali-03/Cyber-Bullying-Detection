import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset (Replace with your actual dataset file)
df = pd.read_csv("datasett.csv")  # Ensure 'your_dataset.csv' exists in your project folder

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)

# Fit the vectorizer on the 'headline' column
X = vectorizer.fit_transform(df['headline'])  # Ensure your dataset has a 'headline' column

# Save the trained TF-IDF Vectorizer
with open("tfidfvectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("TF-IDF Vectorizer trained and saved as tfidfvectorizer.pkl successfully!")

