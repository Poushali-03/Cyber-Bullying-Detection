import pickle

# Load the trained model
with open("LinearSVCTuned.pkl", "rb") as file:
    model = pickle.load(file)

# Test the model with sample input
sample_text = ["You are an idiot!"]
prediction = model.predict(sample_text)

# Print the result
print("Prediction:", "Bullying" if prediction[0] == 1 else "Not Bullying")
