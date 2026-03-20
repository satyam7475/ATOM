"""
ATOM v13 -- One-time enrollment: capture owner face and save encoding.

Run this script to create config/owner_face.npy so ATOM can recognize you (Satyam).
Requires: pip install opencv-python face_recognition numpy

- Camera will open; look at it for 2–3 seconds.
- Press SPACE to capture. Multiple captures improve reliability.
- Press Q to quit and save the encoding.

All data stays on your machine. Nothing is sent to the cloud.
"""

from __future__ import annotations

import sys
from pathlib import Path

def main() -> int:
    try:
        import cv2
        import face_recognition
        import numpy as np
    except ImportError as e:
        print("Install dependencies: pip install opencv-python face_recognition numpy")
        return 1

    config_path = Path(__file__).resolve().parent.parent / "config"
    config_path.mkdir(parents=True, exist_ok=True)
    out_file = config_path / "owner_face.npy"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return 1

    encodings_list = []
    print("Look at the camera. Press SPACE to capture, Q to quit and save.")
    print("Capture 2–3 images for best results.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        if face_encodings:
            cv2.putText(frame, "Face detected - SPACE to capture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face - look at camera", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Captured: {len(encodings_list)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("ATOM Owner Enrollment", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and face_encodings:
            encodings_list.append(face_encodings[0])
            print(f"  Captured {len(encodings_list)}")

    cap.release()
    cv2.destroyAllWindows()

    if not encodings_list:
        print("No captures saved. Run again and press SPACE when your face is detected.")
        return 1

    # Save mean encoding for stability
    encodings_array = np.array(encodings_list)
    mean_encoding = np.mean(encodings_array, axis=0)
    np.save(str(out_file), mean_encoding)
    print(f"Saved owner encoding to {out_file}")
    print("Enable vision in config/settings.json: \"vision\": { \"enabled\": true }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
