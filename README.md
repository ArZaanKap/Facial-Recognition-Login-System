# Facial Recognition Login System ![Face Recognition](https://img.icons8.com/ios/452/face-recognition.png)

## Overview 📖
This project implements a **facial recognition login system**. The system works in two parts:
1. **Encoding Faces**: A script that processes images of faces, extracts facial features, and saves them for future recognition.
2. **Recognizing Faces**: A live webcam interface that detects faces, compares them with saved encodings, and greets recognized individuals with an audio greeting.

---

## What the Code Does 🔍

### 1. **Encoding Faces (`encode_faces.py`)** 📝
   - **Purpose**: This script is responsible for encoding and storing faces for future recognition.
   - **Process**:
     - It loads images from a directory containing images of different people (organized by person name).
     - For each image, it detects faces and crops out the largest detected face.
     - The cropped faces are saved and encoded using the `face_recognition` library.
     - These encodings (which are numerical representations of faces) are saved to a file (`encodings.pickle`) for later use.
   - **Steps**:
     1. Load and resize the images for easier processing.
     2. Detect faces using `face_recognition.face_locations()`.
     3. For each image, encode the detected faces using `face_recognition.face_encodings()`.
     4. Save the encodings and associated names to a `.pickle` file.

### 2. **Recognizing Faces Live (`recognize_live.py`)** 🎥
   - **Purpose**: This script uses the webcam to detect and recognize faces in real time.
   - **Process**:
     - It loads the previously saved face encodings from the `encodings.pickle` file.
     - When the webcam is running, it continuously captures frames, detects faces, and compares them with the saved encodings.
     - If a match is found, it displays the person's name along with a confidence percentage and plays a greeting sound (if available).
   - **Steps**:
     1. Load the saved encodings and corresponding names.
     2. Capture video frames from the webcam and downscale for faster processing.
     3. Detect faces in each frame using `face_recognition.face_locations()`.
     4. Compare each detected face with stored encodings using `face_recognition.face_distance()`.
     5. If a match is found, display the person's name on the screen and play an audio greeting if one exists.

### Key Components 🔑:
- **Face Detection**: Identifies faces in images or webcam frames using the `face_recognition` library.
- **Face Encoding**: Converts faces into numerical representations that can be compared.
- **Audio Greeting**: Plays a greeting sound for recognized individuals.
- **Webcam Interface**: Captures live video from a webcam, processes frames, and identifies recognized faces.

---

## Image File Structure 🗂️

For the **encoding** process to work, your image files should be organized as follows:

Faces as main folder, then sub folders with each person's name, and inside each of those should be images of that person