import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Assuming df is your DataFrame containing the Alzheimer's patient data
# Load your CSV data
df = pd.read_csv('alzeihmers.csv')

# Encoding categorical data ('No' and 'Yes') to numerical values (0 and 1)
df.replace({'No': 0, 'Yes': 1}, inplace=True)

# Specify the order of features
feature_order = ['Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness']
X = df[feature_order]
y = df['Diagnosis of Alzeihmer\'s']

# Perform stratified purposeful sampling with 50% for training and 50% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

# Initialize the DecisionTreeClassifier with entropy criterion
model = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=feature_order, class_names=['No Alzheimer', 'Alzheimer'], filled=True, rounded=True)
plt.show()
