import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("datasett.csv")  # Update with correct file name

# Initialize and train the vectorizer
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
X = vectorizer.fit_transform(df['headline'])

# Train model
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearSVC()
model.fit(X_train, y_train)

# Save vectorizer
with open("tfidfvectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

# Save trained model
with open("LinearSVCTuned.pkl", "wb") as file:
    pickle.dump(model, file)

print("TF-IDF Vectorizer and Model saved successfully!")
