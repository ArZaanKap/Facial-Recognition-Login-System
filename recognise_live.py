# recognize_live.py

import os
import cv2
import pickle
import face_recognition
import numpy as np
from playsound import playsound
import threading

# ─── CONFIG ─────────────────────────────────────────────────────
ENCODINGS_FILE         = "encodings.pickle"
AUDIO_DIR              = "audio"
FRAME_SCALE            = 0.25     # downscale factor (0.25 = ¼ resolution)
PROCESS_EVERY_N_FRAMES = 1        # run detection every 2 frames
TOLERANCE              = 0.45      # max distance for a “match” - LOWER == STRICTER
# ────────────────────────────────────────────────────────────────

# Load cached encodings
if not os.path.exists(ENCODINGS_FILE):
    print(f"[ERROR] '{ENCODINGS_FILE}' not found. Run encode_faces.py first.")
    exit(1)

with open(ENCODINGS_FILE, "rb") as f:
    known_encodings, known_names = pickle.load(f)
print(f"[INFO] Loaded {len(known_encodings)} known faces")


# Build mapping from name -> audio file
audio_files = {}
for fname in os.listdir(AUDIO_DIR):
    name, ext = os.path.splitext(fname)
    if ext.lower() in ('.wav', '.mp3', '.ogg'):
        audio_files[name] = os.path.join(AUDIO_DIR, fname)

# Track who has been greeted to avoid repeats
greeted = set()


# Start webcam
video = cv2.VideoCapture(0)
frame_count = 0

# Keep last‑seen so we don’t flicker
prev_locs, prev_names, prev_confs = [], [], []

print("[INFO] Starting webcam (press 'q' to quit)…")
while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1
    face_locations = []
    face_names     = []
    face_confs     = []

    # ── Only run heavy detection every N frames ───────────────
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # 1) Resize for speed
        small = cv2.resize(frame, (0,0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # 2) Detect & encode
        locs = face_recognition.face_locations(rgb)                     # uses fast HOG
        encodings = face_recognition.face_encodings(rgb, locs)

        # 3) Compare & compute confidences
        for encoding in encodings:
            distances = face_recognition.face_distance(known_encodings, encoding)

            if len(distances) > 0:
                best_idx  = np.argmin(distances)
                best_dist = distances[best_idx]

                if best_dist <= TOLERANCE:
                    name      = known_names[best_idx]
                    # Map distance→confidence%: closer=100%
                    conf = max(0.0, (1 - best_dist / TOLERANCE) * 100)
                else:
                    name, conf = "Unknown", 0.0
            else:
                name, conf = "Unknown", 0.0

            face_names.append(name)
            face_confs.append(conf)

        # Only update if we found at least one face
        if locs:
            prev_locs, prev_names, prev_confs = locs, face_names, face_confs
        else:
            # else: keep last seen
            prev_locs, prev_names, prev_confs = [], [], []

    # between detections reuse last results
    face_locations, face_names, face_confs = prev_locs, prev_names, prev_confs



    # Draw results
    for (top, right, bottom, left), name, conf in zip(face_locations, face_names, face_confs):
        # scale coords back up
        top    = int(top    / FRAME_SCALE)
        right  = int(right  / FRAME_SCALE)
        bottom = int(bottom / FRAME_SCALE)
        left   = int(left   / FRAME_SCALE)

        # box + label with % certainty
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        text = f"{name} ({conf:.0f}%)"
        cv2.putText(frame, text, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        

        # Play greeting if we have an audio file and haven't greeted yet
        if name != "Unknown" and f"{name}_greeting" in audio_files and name not in greeted:
            def play(path):
                try:
                    playsound(path)
                except Exception as e:
                    print(f"[WARNING] Could not play audio for {name}: {e}")

            # run greeting in separate thread to prevent video freeze
            threading.Thread(target=play, args=(audio_files[f"{name}_greeting"],), daemon=True).start()
            greeted.add(name)


    cv2.imshow("Family Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
