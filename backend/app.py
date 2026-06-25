import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, static_folder='../frontend', static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Pretrained Haar cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Simple recognition database using LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer_read = False
labels = {}

# Load sample faces for recognition if a model file exists
MODEL_FILE = 'face_recognizer.yml'
LABELS_FILE = 'labels.npy'

if os.path.exists(MODEL_FILE) and os.path.exists(LABELS_FILE):
    recognizer.read(MODEL_FILE)
    labels = np.load(LABELS_FILE, allow_pickle=True).item()
    recognizer_read = True


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/detect', methods=['POST'])
def detect_faces():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)

        image = cv2.imread(filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        # allow skipping recognition and only doing detection
        detect_only = False
        if 'detect_only' in request.form:
            val = request.form.get('detect_only')
            if val is not None and str(val).lower() in ('1', 'true', 'yes', 'on'):
                detect_only = True

        detections = []
        # decide per-request whether to run recognition
        do_recognition = recognizer_read and not detect_only

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            name = 'Unknown'
            confidence = None
            if do_recognition:
                try:
                    label_id, confidence = recognizer.predict(face_roi)
                    if confidence < 80:
                        name = labels.get(label_id, 'Unknown')
                    else:
                        name = 'Unknown'
                except Exception:
                    # if recognition fails for any reason, mark Unknown
                    name = 'Unknown'
            detections.append({
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h),
                'name': name,
                'confidence': float(confidence) if confidence is not None else None
            })

        return jsonify({'detections': detections})

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/api/train', methods=['POST'])
def train_recognizer():
    images = []
    labels_list = []
    assigned_label = 0

    for label_name, image_files in request.files.lists():
        for image_file in image_files:
            if image_file and allowed_file(image_file.filename):
                filename = secure_filename(image_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(filepath)
                image = cv2.imread(filepath)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
                if len(faces) == 0:
                    continue
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                images.append(face_roi)
                labels_list.append(assigned_label)
        labels[label_name] = assigned_label
        assigned_label += 1

    if len(images) == 0:
        return jsonify({'error': 'No faces found for training'}), 400

    recognizer.train(images, np.array(labels_list))
    recognizer.save(MODEL_FILE)
    np.save(LABELS_FILE, labels)

    return jsonify({'message': 'Training complete', 'labels': labels})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
