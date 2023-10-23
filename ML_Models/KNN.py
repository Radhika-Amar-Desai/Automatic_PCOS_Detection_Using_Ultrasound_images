import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the directory where your dataset is located
dataset_dir = 'Processed_images'

# Initialize lists to store image data and corresponding labels
X = []  # Image data
y = []  # Labels

# Load and preprocess images
for class_folder in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_folder)
    if os.path.isdir(class_path):
        label = class_folder  # Use the folder name as the label
        for image_file in os.listdir(class_path):
            if image_file.endswith(".jpg") or image_file.endswith(".jpeg") or image_file.endswith(".png"):  # Assuming your images are in JPG format
                image_path = os.path.join(class_path, image_file)
                image = cv2.imread(image_path)
                # Preprocess the image (e.g., resize to a fixed size)
                # You can also perform other preprocessing steps here
                image = cv2.resize(image, (64, 64))  # Adjust the size as needed
                X.append(image)
                y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the k-NN classifier
k = 2  # Choose an appropriate value for k
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train.reshape(len(X_train), -1), y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test.reshape(len(X_test), -1))

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
