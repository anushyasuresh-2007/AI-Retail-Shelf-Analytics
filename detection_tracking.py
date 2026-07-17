from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Input video
video_path = "videos/Copy of Alif Store-Viewing-10.mp4"

# Detect and track people
results = model.track(
    source=video_path,
    classes=[0],          # 0 = person
    tracker="bytetrack.yaml",
    persist=True,
    save=True,
    show=True
)