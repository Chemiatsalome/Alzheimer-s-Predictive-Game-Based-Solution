import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Sample data as provided
df = pd.read_csv('alzeihmers.csv')

# Display basic information about the dataset
print(df.info())
print(df.head())

# Check the distribution of the target variable
print(df['Diagnosis'].value_counts())

# Extract features and target
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

# Create separate dataframes for each class
class_0 = df[df['Diagnosis'] == 0]
class_1 = df[df['Diagnosis'] == 1]

# Number of samples from each class (proportionate to the original dataset)
n_samples_class_0 = int(20 * len(class_0) / len(df))
n_samples_class_1 = 20 - n_samples_class_0  # remaining samples

# Sample from each class
sampled_class_0 = class_0.sample(n=n_samples_class_0, random_state=42)
sampled_class_1 = class_1.sample(n=n_samples_class_1, random_state=42)

# Combine the samples
stratified_sample = pd.concat([sampled_class_0, sampled_class_1])

# Separate features and target for the stratified sample
X_20_samples = stratified_sample.drop(columns=['Diagnosis'])
y_20_samples = stratified_sample['Diagnosis']

# Split into final training and testing sets (10 samples each)
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_20_samples, y_20_samples, test_size=0.5, stratify=y_20_samples, random_state=42)

# Verify the split
print("Training set distribution:\n", y_train_final.value_counts())
print("Testing set distribution:\n", y_test_final.value_counts())



# Initialize the classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train_final, y_train_final)

# Predict on the test set
y_pred = clf.predict(X_test_final)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test_final, y_pred))
print("Classification Report:\n", classification_report(y_test_final, y_pred))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Plot the decision tree
# Increase DPI for better resolution
plt.figure(figsize=(20, 10), dpi=100)

# Plot the decision tree
plot_tree(
    clf, 
    feature_names=X_train_final.columns, 
    class_names=['No Alzheimer', 'Alzheimer'], 
    filled=True, 
    rounded=True, 
    fontsize=12,  # Increase the font size for better readability
    precision=2  # Set the precision for floating point numbers in the output
)

# Save the plot as an image
plt.savefig("decision_tree.png", format='png', bbox_inches='tight')
plt.show()
