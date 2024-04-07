import cv2
import os
import numpy as np

# Function to create directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Initialize variables
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'database'
images = []
labels = []

# Create 'database' directory if it doesn't exist
create_directory(datasets)

# Load face cascade
face_cascade = cv2.CascadeClassifier(haar_file)

# Read face images and labels
for subdir, _, files in os.walk(datasets):
    for file in files:
        path = os.path.join(subdir, file)
        label = os.path.basename(subdir)
        image = cv2.imread(path, 0)  # Read image in grayscale
        if image is not None:
            images.append(image)
            labels.append(int(label))

# Ensure images and labels are numpy arrays
images = np.array(images)
labels = np.array(labels)

# Initialize LBPH face recognizer model
if cv2.__version__.startswith('3'):
    model = cv2.face.createLBPHFaceRecognizer()
else:
    model = cv2.face.LBPHFaceRecognizer_create()

# Train the model
model.train(images, labels)
