import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset (replace 'path_to_dataset.csv' with your actual file path)
df = pd.read_csv('alzeihmers2.csv')

# Display basic information about the dataset
print(df.info())
print(df.describe())
print(df.head())


# Now selecting 20 samples 
# Separate data by class
class_0 = df[df['Diagnosis'] == 0]
class_1 = df[df['Diagnosis'] == 1]

# Determine the number of samples for each class
n_samples_class_0 = int(20 * len(class_0) / len(df))
n_samples_class_1 = 20 - n_samples_class_0  # Remaining samples

# Sample from each class
sampled_class_0 = class_0.sample(n=n_samples_class_0, random_state=42)
sampled_class_1 = class_1.sample(n=n_samples_class_1, random_state=42)

# Combine the samples
stratified_sample = pd.concat([sampled_class_0, sampled_class_1])

# Define the features and target
features = ['CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE']
X_selected = stratified_sample[features]
y_selected = stratified_sample['Diagnosis']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.5, stratify=y_selected, random_state=42)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can tune the number of neighbors

# Train the classifier
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Alzheimer', 'Alzheimer'], 
            yticklabels=['No Alzheimer', 'Alzheimer'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()