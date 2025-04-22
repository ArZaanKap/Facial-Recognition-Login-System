# encode_faces.py

import os
import face_recognition
import numpy as np
from PIL import Image
import pickle
import shutil

FACES_DIR = "faces"
ENCODINGS_FILE = "encodings.pickle"
MAX_IMG_SIZE = 1600
CROPS_DIR = "training_faces"


def load_and_resize(image_path, max_size=MAX_IMG_SIZE):
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))
    return np.array(img)

def encode_faces():
    known_encodings = []
    known_names = []

     # Clear previous cropped faces
    if os.path.exists(CROPS_DIR):
        shutil.rmtree(CROPS_DIR)
    os.makedirs(CROPS_DIR, exist_ok=True)  ## MAKE CROPPED FACES DIR

    print("[INFO] Encoding known faces...")

    for person_name in os.listdir(FACES_DIR):
        person_folder = os.path.join(FACES_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue

        total_imgs = 0
        valid_imgs = 0

        # make a crop folder per person
        crop_person = os.path.join(CROPS_DIR, f"{person_name}_cropped")
        os.makedirs(crop_person, exist_ok=True)

        for img_name in os.listdir(person_folder):

            total_imgs += 1

            img_path = os.path.join(person_folder, img_name)

            try:
                img = load_and_resize(img_path)


                # detect all faces
                locs = face_recognition.face_locations(img)
                if not locs:
                    print(f"[!] No face found in {img_path}, skipping.")
                    continue

                valid_imgs += 1

                # pick the largest face by area
                areas = [(b - t) * (r - l) for (t, r, b, l) in locs]
                max_idx = int(np.argmax(areas))
                top, right, bottom, left = locs[max_idx]

                # crop & save that face
                crop = img[top:bottom, left:right]
                crop_filename = f"{person_name}_{valid_imgs}.jpg"
                crop_path = os.path.join(crop_person, crop_filename)
                Image.fromarray(crop).save(crop_path)

                # encode only that face
                encodings = face_recognition.face_encodings(img, [locs[max_idx]])

                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                    print(f"[+] Encoded {img_name} for {person_name}")

                else:
                    print(f"[!] No face found in {img_path}, skipping.")

            except Exception as e:
                print(f"[!] Error loading {img_path}: {e}")

        print()
        print(f"✔️ Loaded {valid_imgs}/{total_imgs} valid face(s) for '{person_name}'")
        print()


    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((known_encodings, known_names), f)

    print(f"[DONE] Saved {len(known_encodings)} encodings to {ENCODINGS_FILE}")



if __name__ == "__main__":

    encode_faces()
