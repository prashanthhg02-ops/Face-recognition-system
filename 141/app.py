from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
FACES_DIR = DATA_DIR / "faces"
MODEL_PATH = DATA_DIR / "face_model.yml"
LABELS_PATH = DATA_DIR / "labels.json"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def ensure_directories() -> None:
    FACES_DIR.mkdir(parents=True, exist_ok=True)


def get_detector() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("OpenCV could not load its bundled face detector.")
    return detector


def open_camera(index: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {index}. Check camera permissions or --camera.")
    return camera


def register_person(name: str, camera_index: int, samples: int) -> None:
    ensure_directories()
    person_dir = FACES_DIR / name.strip().replace(" ", "_")
    person_dir.mkdir(parents=True, exist_ok=True)
    detector = get_detector()
    camera = open_camera(camera_index)
    captured = 0

    print(f"Registering {name}. Look at the camera; press q to cancel.")
    try:
        while captured < samples:
            ok, frame = camera.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))
            for x, y, width, height in faces:
                face = gray[y : y + height, x : x + width]
                output = person_dir / f"{captured + 1:04d}.jpg"
                cv2.imwrite(str(output), face)
                captured += 1
                cv2.rectangle(frame, (x, y), (x + width, y + height), (64, 210, 120), 2)
                cv2.putText(frame, f"Captured {captured}/{samples}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64, 210, 120), 2)
                break
            cv2.imshow("Register face", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Saved {captured} samples in {person_dir}")


def train_model() -> None:
    ensure_directories()
    detector = get_detector()
    faces: list[object] = []
    labels: list[int] = []
    label_names: dict[str, str] = {}

    for label, person_dir in enumerate(sorted(path for path in FACES_DIR.iterdir() if path.is_dir())):
        images = sorted(person_dir.glob("*.jpg"))
        if not images:
            continue
        label_names[str(label)] = person_dir.name.replace("_", " ")
        for image_path in images:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            detected = detector.detectMultiScale(image, 1.1, 4)
            face = image
            if len(detected):
                x, y, width, height = max(detected, key=lambda box: box[2] * box[3])
                face = image[y : y + height, x : x + width]
            faces.append(face)
            labels.append(label)

    if not faces:
        raise RuntimeError("No face samples found. Run `python app.py register --name YourName` first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, __import__("numpy").array(labels))
    recognizer.write(str(MODEL_PATH))
    LABELS_PATH.write_text(json.dumps(label_names, indent=2), encoding="utf-8")
    print(f"Trained on {len(faces)} samples for {len(label_names)} people.")


def recognize(camera_index: int, threshold: float) -> None:
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise RuntimeError("Model not found. Register a person and run `python app.py train` first.")
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    detector = get_detector()
    camera = open_camera(camera_index)

    print("Recognition is running. Press q to stop.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for x, y, width, height in detector.detectMultiScale(gray, 1.2, 5, minSize=(100, 100)):
                label, confidence = recognizer.predict(gray[y : y + height, x : x + width])
                name = labels.get(str(label), "Unknown") if confidence <= threshold else "Unknown"
                color = (64, 210, 120) if name != "Unknown" else (70, 90, 230)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(frame, f"{name} | {confidence:.0f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Face recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Local face recognition system using OpenCV ML.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register", help="Capture face samples for one person")
    register.add_argument("--name", required=True, help="Person name")
    register.add_argument("--camera", type=int, default=0)
    register.add_argument("--samples", type=int, default=30)
    subparsers.add_parser("train", help="Train the recognizer from saved samples")
    recognize_parser = subparsers.add_parser("recognize", help="Start live recognition")
    recognize_parser.add_argument("--camera", type=int, default=0)
    recognize_parser.add_argument("--threshold", type=float, default=75.0, help="Lower values are stricter")
    args = parser.parse_args()

    try:
        if args.command == "register":
            register_person(args.name, args.camera, args.samples)
        elif args.command == "train":
            train_model()
        else:
            recognize(args.camera, args.threshold)
    except (RuntimeError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
