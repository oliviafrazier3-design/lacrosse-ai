import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import deque

# 1. Load Models
player_model = YOLO('player_model.pt')
jersey_model = YOLO('jersey_model.pt')

# --- SETTINGS ---
CALIBRATION_LIMIT = 120
calibration_buffer = []
team_centers = None
player_data = {}
best_num_conf = {}

def get_torso_color(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1
    torso = frame[y1 + int(h*0.35):y1 + int(h*0.55), x1 + int(w*0.3):x1 + int(w*0.7)]
    if torso.size == 0: return None
    
    pixels = torso.reshape(-1, 3)
    # Aggressive grass filter
    mask = (pixels[:, 1] < pixels[:, 0] + 15) | (pixels[:, 1] < pixels[:, 2] + 15)
    valid_pixels = pixels[mask]
    
    if len(valid_pixels) < 5: return None
    return np.mean(valid_pixels, axis=0)

# --- MAIN LOOP ---
cap = cv2.VideoCapture('test_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = player_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5, iou=0.6, verbose=False)
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Step 1: Calibration Gathering
        if team_centers is None:
            for box in boxes:
                color = get_torso_color(frame, box)
                if color is not None: calibration_buffer.append(color)
            if len(calibration_buffer) >= CALIBRATION_LIMIT:
                kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(calibration_buffer)
                team_centers = kmeans.cluster_centers_

        # Step 2: Identification & Continuous Re-Verification
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            if track_id not in player_data:
                # Use a deque to keep a rolling window of the last 100 team votes
                player_data[track_id] = {'team': 'identifying', 'num': '', 'votes': deque(maxlen=100)}

            # Re-check color every frame (or every 5 frames for speed)
            if team_centers is not None:
                current_color = get_torso_color(frame, box)
                if current_color is not None:
                    dist0 = np.linalg.norm(current_color - team_centers[0])
                    dist1 = np.linalg.norm(current_color - team_centers[1])
                    
                    # Add current observation to the rolling vote
                    obs = "Team A" if dist0 < dist1 else "Team B"
                    player_data[track_id]['votes'].append(obs)
                    
                    # Update the team label based on the MAJORITY of the rolling window
                    if len(player_data[track_id]['votes']) >= 20:
                        v_list = list(player_data[track_id]['votes'])
                        player_data[track_id]['team'] = max(set(v_list), key=v_list.count)

            # Jersey Number Logic (Same as before)
            crop = frame[max(0, y1-12):y2+12, max(0, x1-12):x2+12]
            if crop.size > 0:
                num_res = jersey_model(crop, conf=0.3, verbose=False)
                if len(num_res[0].boxes) > 0:
                    current_conf = num_res[0].boxes.conf.mean().item()
                    digits = []
                    for b in num_res[0].boxes:
                        digits.append({'x': b.xyxy[0][0].item(), 'v': jersey_model.names[int(b.cls[0])]})
                    digits.sort(key=lambda d: d['x'])
                    combined = "".join([d['v'] for d in digits])
                    
                    if len(combined) <= 2:
                        prev_num = player_data[track_id]['num']
                        if len(combined) > len(prev_num) or (len(combined) == len(prev_num) and current_conf > best_num_conf.get(track_id, 0)):
                            player_data[track_id]['num'] = combined
                            best_num_conf[track_id] = current_conf

            # Step 3: Drawing
            team = player_data[track_id]['team']
            num = player_data[track_id]['num']
            label = f"{team} #{num}"
            color = (0, 255, 255) if team == "Team A" else (255, 255, 255) if team == "Team B" else (200, 200, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Lacrosse AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
