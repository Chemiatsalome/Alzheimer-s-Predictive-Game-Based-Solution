from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Create the Flask app
app = Flask(__name__)

# Sample data as provided (you can load your trained model here instead)
df = pd.read_csv('alzeihmers.csv')

# Extract features and target
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

# Train the classifier with the entire dataset (for simplicity in this example)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    symptoms = request.form.to_dict()
    
    # Convert form data to DataFrame
    data = pd.DataFrame([symptoms])
    
    # Ensure the data types are correct
    data = data.astype(int)
    
    # Predict using the model
    prediction = clf.predict(data)[0]
    
    # Map the prediction to a human-readable label
    if prediction == 0:
        result = "No Alzheimer"
    else:
        result = "Alzheimer"
    
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
