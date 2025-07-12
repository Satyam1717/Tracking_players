import cv2
from ultralytics import YOLO
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import torchreid
import torchvision  
from torchvision import transforms
import torch
# Initialize the ReID model
model_reid = torchreid.models.build_model(
    name='osnet_x0_25',  # Lightweight and fast
    num_classes=1000,
    pretrained=True
)
model_reid.eval()
model_reid.cuda()  # Remove this line if you do not have a GPU



# Define preprocessing for ReID model
reid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_reid_embedding(image, bbox):
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = reid_transform(crop)
    crop = crop.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = model_reid(crop.cuda()).cpu().numpy().flatten()
    return embedding

#  # Function to extract appearance features using ReID model
# def extract_appearance_features(image, bbox):
#     x1, y1, x2, y2 = bbox
#     crop = image[y1:y2, x1:x2]
#     if crop.size == 0:
#         return None
#     crop = cv2.resize(crop, (256, 128))  # Resize to fit ReID model input
#     crop = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0).cuda()  # Convert to tensor and move to GPU
#     with torch.no_grad():
#         features = model_reid(crop)
#     return features.cpu().numpy().flatten()  # Move back to CPU and flatten the array


# Function to extract color histogram for appearance features
def get_color_histogram(image, bbox):
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Function to update appearance history for each track
def update_appearance_history(track_id, frame, bbox):
    if track_id not in track_appearances:
        track_appearances[track_id] = []
    hist = get_color_histogram(frame, bbox)
    if hist is not None:
        track_appearances[track_id].append(hist)
        if len(track_appearances[track_id]) > 5:  # Keep last 5 appearances
            track_appearances[track_id].pop(0)

# Load YOLOv8 model
model = YOLO('best.pt')

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=10)  # You may adjust max_age as needed

# Dictionary to store appearance histories
track_appearances = {}

# Open video file
cap = cv2.VideoCapture('15sec_input_720p.mp4')  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    bbs = []

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if cls == 2 and conf > 0.6:  # Only 'player'
                w, h = x2 - x1, y2 - y1
                # Try deep ReID embedding
                embedding = get_reid_embedding(frame, (x1, y1, x2, y2))
                if embedding is not None:
                    feature = embedding
                else:
                    # Fallback to color histogram
                    hist = get_color_histogram(frame, (x1, y1, x2, y2))
                    if hist is not None:
                        feature = hist
                    else:
                        continue  # Skip if both fail
                # Append bounding box, confidence, class, and feature (always 4 elements)
                bbs.append(([x1, y1, w, h], conf, cls, feature))


    # Update tracker (DeepSORT uses appearance features internally if configured)
    tracks = tracker.update_tracks(bbs, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        if track.age < 3:  # Skip tracks that are too new
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        # Update appearance history for this track
        update_appearance_history(track_id, frame, (x1, y1, x2, y2))
        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Player Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
