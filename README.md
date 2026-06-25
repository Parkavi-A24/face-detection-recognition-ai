# Face Detection and Recognition AI

## Project Overview

Face Detection and Recognition AI is a Computer Vision application that automatically detects and identifies human faces in images. The system uses OpenCV's Haar Cascade Classifier to locate faces and can be extended with advanced face recognition techniques such as Siamese Networks, FaceNet, or ArcFace.

The application provides a web-based interface where users can upload images and receive real-time face detection results.

---

## Features

* Upload image from browser
* Automatic face detection
* Face counting functionality
* Flask REST API backend
* OpenCV-based image processing
* User-friendly interface

---

## Technologies Used

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Python
* Flask
* Flask-CORS

### AI / Computer Vision

* OpenCV
* Haar Cascade Classifier
* NumPy

---

## Project Structure

face-detection-recognition/

├── backend/

│ ├── app.py

│ ├── face_detector.py

│ ├── requirements.txt

│ └── uploads/

├── frontend/

│ ├── index.html

│ ├── style.css

│ └── script.js

├── README.md

├── LICENSE

└── .gitignore

---

## Installation

1. Clone the repository

2. Install dependencies

pip install -r requirements.txt

3. Run backend

python app.py

4. Open frontend/index.html

5. Upload image and detect faces

---

## Workflow

1. User uploads image.
2. Image is sent to Flask backend.
3. OpenCV processes image.
4. Haar Cascade detects faces.
5. Number of detected faces is returned.
6. Result is displayed on the webpage.

---

## Future Enhancements

* Face Recognition using ArcFace
* Real-time Webcam Detection
* Attendance Management System
* Face Mask Detection
* Emotion Recognition

---

## Author

parkavi A
BE COMPUTER SCIENCE AND ENGINEERING (AIML)
