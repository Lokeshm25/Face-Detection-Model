#!pip install mtcnn keras-facenet opencv-python matplotlib scikit-learn

import cv2 as cv
import numpy as np
import pickle
import os
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Path to trained SVM model
model_path = r"C:\Users\Lenovo\OneDrive\Desktop\Hack\New folder\Python\svm_model_160x160.pkl"

# Load SVM model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Path to saved LabelEncoder
encoder_path = r"C:\Users\Lenovo\OneDrive\Desktop\Hack\New folder\Python\label_encoder.pkl"

# Load the saved LabelEncoder
with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)

print("✅ LabelEncoder loaded successfully!")
# Initialize FaceNet for embeddings
embedder = FaceNet()

# Initialize MTCNN for face detection
detector = MTCNN()

print("✅ Model and dependencies loaded successfully!")


def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]


def recognize_face(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    faces = detector.detect_faces(img)
    if not faces:
        print("❌ No face detected!")
        return None

    x, y, w, h = faces[0]['box']
    t_img = img[y:y+h, x:x+w]
    t_img = cv.resize(t_img, (160, 160))

    # Get FaceNet embedding
    face_embedding = get_embedding(t_img)

    # Predict using SVM model
    prediction = model.predict([face_embedding])
    predicted_label = encoder.inverse_transform(prediction)[0]
    # Get label

    # Draw the result on the image
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
    cv.putText(img, str(predicted_label), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the result
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    return predicted_label


def find_person_in_group(image_path, target_name):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    faces = detector.detect_faces(img)
    if not faces:
        print("❌ No faces detected in the image!")
        return None

    found = False
    for face in faces:
        x, y, w, h = face['box']
        detected_face = img[y:y+h, x:x+w]
        detected_face = cv.resize(detected_face, (160, 160))

        # Get embedding and predict
        face_embedding = get_embedding(detected_face)

        # Predict using SVM model
        prediction = model.predict([face_embedding])
        predicted_label = encoder.inverse_transform(prediction)[0]

        # Check if the detected person matches the input name
        if predicted_label == target_name:
            found = True
            print(f"✅ {target_name} found in the photo!")
        else:
            continue

        # Draw bounding box
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
        cv.putText(img, str(predicted_label), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        break


    # Show result
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    if not found:
        print(f"❌ {target_name} not found in the image.")


def recognize_live_video():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)

        for face in faces:
            x, y, w, h = face['box']
            detected_face = img_rgb[y:y+h, x:x+w]
            detected_face = cv.resize(detected_face, (160, 160))

            # Get embedding and predict
            embedding = get_embedding(detected_face)
            prediction = model.predict([embedding])
            predicted_label = encoder.inverse_transform(prediction)[0]

            # Draw bounding box
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
            cv.putText(frame, str(predicted_label), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow("Live Face Recognition", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


while True:
    print("\n🎯 Face Recognition System")
    print("1️⃣ Identify a person's name from an image")
    print("2️⃣ Identify a specific person in a group photo")
    print("3️⃣ Live video face recognition")
    print("4️⃣ Exit")

    choice = input("Enter your choice (1/2/3/4): ")

    if choice == '1':
        img_path = input("Enter the image path: ")
        recognized_person = recognize_face(img_path)
        if recognized_person:
            print(f"🎉 Recognized Person: {recognized_person}")

    elif choice == '2':
        img_path = input("Enter the group photo path: ")
        person_name = input("Enter the person's name to search: ")
        find_person_in_group(img_path, person_name)

    elif choice == '3':
        print("🔴 Starting live video face recognition... Press 'q' to exit.")
        recognize_live_video()

    elif choice == '4':
        print("🚪 Exiting Face Recognition System.")
        break

    else:
        print("❌ Invalid choice. Please try again.")


# import cv2

# cap = cv2.VideoCapture(0)  # Try 1 if 0 doesn't work

# if not cap.isOpened():
#     print("❌ Error: Could not open webcam. Try using cv2.VideoCapture(1).")
# else:
#     print("✅ Webcam is working!")

# cap.release()
# cv2.destroyAllWindows()
