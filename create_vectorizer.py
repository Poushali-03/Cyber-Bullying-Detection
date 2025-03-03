import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample dataset (replace with your actual dataset)
documents = [
    "Cyberbullying is harmful and should be stopped",
    "Online harassment is a serious problem",
    "Protect yourself from cyber threats",
    "Be aware of digital safety",
    "Social media can be both positive and negative"
]

# Create a TF-IDF Vectorizer and fit on the sample data
vectorizer = TfidfVectorizer(stop_words="english")
vectorizer.fit(documents)

# Save the trained vectorizer vocabulary
with open("tfidfvectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer.vocabulary_, file)

print("✅ tfidfvectorizer.pkl has been successfully created!")
