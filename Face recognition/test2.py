import cv2
import numpy as np
import os

# Constants
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'database'
width, height = 130, 100

try:
    # Create lists of images and corresponding labels
    (images, labels, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            names[id] = subdir
            subject_path = os.path.join(datasets, subdir)
            for filename in os.listdir(subject_path):
                path = os.path.join(subject_path, filename)
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1

    # Convert lists to numpy arrays
    (images, labels) = [np.array(lis) for lis in [images, labels]]

    # Initialize EigenFaceRecognizer model
    model = cv2.face.EigenFaceRecognizer_create()

    # Train the model
    model.train(images, labels)

    # Initialize face cascade classifier
    face_cascade = cv2.CascadeClassifier(haar_file)

    # Initialize webcam
    webcam = cv2.VideoCapture(0)

    # Main loop for face recognition
    while True:
        (ret, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            # Try to recognize the face
            prediction = model.predict(face_resize.flatten())
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Calculate confidence level as percentage
            confidence = 100 - prediction[1]

            if confidence > 50:  # Only consider high-confidence predictions
                cv2.putText(im, '%s - %.2f%%' % (names[prediction[0]], confidence), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            else:
                cv2.putText(im, 'Not Recognized', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        cv2.imshow('OpenCV', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close windows
    webcam.release()
    cv2.destroyAllWindows()

except Exception as e:
    print("An error occurred:", e)
