import numpy as np
import json
import cv2
import os
from django.shortcuts import render, redirect
from django.http import HttpResponse
from PIL import Image

def create_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_face_id(directory: str) -> int:
    user_ids = []
    for filename in os.listdir(directory):
        number = int(os.path.split(filename)[-1].split("-")[1])
        user_ids.append(number)
    user_ids = sorted(list(set(user_ids)))
    max_user_ids = 1 if len(user_ids) == 0 else max(user_ids) + 1
    for i in sorted(range(0, max_user_ids)):
        try:
            if user_ids.index(i):
                face_id = i
        except ValueError:
            return i
    return max_user_ids

def save_name(face_id: int, face_name: str, filename: str) -> None:
    names_json = {}
    if os.path.exists(filename):
        with open(filename, 'r') as fs:
            names_json = json.load(fs)
    names_json[face_id] = face_name
    with open(filename, 'w') as fs:
        json.dump(names_json, fs, ensure_ascii=False, indent=4)

def capture_face(request):
    if request.method == 'POST':
        face_name = request.POST.get('face_name')
        
        directory = 'images'
        cascade_classifier_filename = 'haarcascade_frontalface_default.xml'
        names_json_filename = 'names.json'
        
        # Create 'images' directory if it doesn't exist
        create_directory(directory)
        
        # Load the pre-trained face cascade classifier
        faceCascade = cv2.CascadeClassifier(cascade_classifier_filename)
        
        # Open a connection to the default camera
        cam = cv2.VideoCapture(0)
        
        # Set camera dimensions
        cam.set(3, 640)
        cam.set(4, 480)
        
        count = 0
        face_id = get_face_id(directory)
        save_name(face_id, face_name, names_json_filename)
        
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                count += 1
                cv2.imwrite(f'./images/Users-{face_id}-{count}.jpg', gray[y:y+h, x:x+w])
                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff
            if k < 30 or count >= 30:
                break
        
        cam.release()
        cv2.destroyAllWindows()
        return HttpResponse("Face capture completed successfully.")

    return render(request, 'face_recognition/capture_face.html')



def train_faces(request):
    # Directory path where the face images are stored.
    path = './images/'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("\n[INFO] Training...")
    # Haar cascade file for face detection
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    def get_images_and_labels(path):
        """
        Load face images and corresponding labels from the given directory path.
    
        Parameters:
            path (str): Directory path containing face images.
    
        Returns:
            list: List of face samples.
            list: List of corresponding labels.
        """
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []
    
        for image_path in image_paths:
            # Convert image to grayscale
            PIL_img = Image.open(image_path).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            
            # Extract the user ID from the image file name
            id = int(os.path.split(image_path)[-1].split("-")[1])
    
            # Detect faces in the grayscale image
            faces = detector.detectMultiScale(img_numpy)
    
            for (x, y, w, h) in faces:
                # Extract face region and append to the samples
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
    
        return face_samples, ids
    
    faces, ids = get_images_and_labels(path)
    
    if len(faces) == 0:
        return HttpResponse("No faces found for training.")
    
    # Train the recognizer with the face samples and corresponding labels
    recognizer.train(faces, np.array(ids))
    
    # Save the trained model into the current directory
    recognizer.write('trainer.yml')
    
    print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    
    return HttpResponse(f"{len(np.unique(ids))} faces trained successfully.")

def recognize_faces(request):
    if request.method == 'POST':
        # Create LBPH Face Recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        # Load the trained model
        recognizer.read('trainer.yml')

        # Path to the Haar cascade file for face detection
        face_cascade_path = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(face_cascade_path)

        # Load names from JSON
        with open('names.json', 'r') as fs:
            names = json.load(fs)
            names = list(names.values())

        # Video Capture from the default camera
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # Set width
        cam.set(4, 480)  # Set height

        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                if confidence > 70:
                    try:
                        name = names[id]
                        confidence = "  {0}%".format(round(confidence))
                    except IndexError:
                        name = "Who are you?"
                        confidence = "N/A"
                else:
                    name = "Who are you?"
                    confidence = "N/A"

                cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, confidence, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

            cv2.imshow('camera', img)

            # Break the loop on Escape key press
            if cv2.waitKey(10) & 0xff == 27:
                break

        cam.release()
        cv2.destroyAllWindows()
        return HttpResponse("Face recognition completed.")

    return render(request, 'face_recognition/recognize_faces.html')

