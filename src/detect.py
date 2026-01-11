from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans

# 1. Load model
model = YOLO("player_model.pt")

# --- STORAGE ---
track_hist_data = {}
player_teams = {}
basic_colors = None
t1_idx_list = None
t2_idx_list = None

def create_color_histogram_nearest(image, current_basic_colors):
    h, w, _ = image.shape
    # Focus on the Torso
    torso = image[int(h*0.35):int(h*0.65), int(w*0.3):int(w*0.7)]
    
    if torso.size == 0:
        return np.zeros(len(current_basic_colors))

    colors_np = np.array(current_basic_colors)
    num_colors = len(colors_np)
    pixels = torso.reshape(-1, 3)
    
    # Vectorized distance calculation
    distances = np.sum((pixels[:, np.newaxis, :] - colors_np[np.newaxis, :, :]) ** 2, axis=2)
    assignments = np.argmin(distances, axis=1)
    
    return np.bincount(assignments, minlength=num_colors)

def auto_calibrate_teams(frame, boxes):
    """Automatically finds team profiles, specifically tuned to detect 'White' vs 'Color'."""
    player_averages = []
    all_pixels = []
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        h, w = y2-y1, x2-x1
        torso = frame[y1+int(h*0.3):y1+int(h*0.7), x1+int(w*0.2):x1+int(w*0.8)]
        if torso.size > 0:
            avg_color = torso.mean(axis=(0, 1))
            player_averages.append(avg_color)
            all_pixels.append(torso.reshape(-1, 3))

    if len(player_averages) < 2:
        return None, None, None

    # 1. Cluster all torso pixels to get the palette
    all_pixels = np.vstack(all_pixels)
    kmeans_colors = KMeans(n_clusters=6, random_state=42, n_init=10).fit(all_pixels)
    found_basic_colors = kmeans_colors.cluster_centers_.astype(int)

    # 2. Separate players into two teams
    kmeans_teams = KMeans(n_clusters=2, random_state=42, n_init=10).fit(player_averages)
    
    team1_indices = []
    team2_indices = []
    
    for i, b_color in enumerate(found_basic_colors):
        dist_to_t1 = np.linalg.norm(b_color - kmeans_teams.cluster_centers_[0])
        dist_to_t2 = np.linalg.norm(b_color - kmeans_teams.cluster_centers_[1])
        if dist_to_t1 < dist_to_t2:
            team1_indices.append(i)
        else:
            team2_indices.append(i)
            
    return found_basic_colors, team1_indices, team2_indices

def assign_team(hist, t1_idx, t2_idx):
    """Refined logic: Lower thresholds to ensure 'White' team is picked up."""
    t1_sum = sum(hist[i] for i in t1_idx)
    t2_sum = sum(hist[i] for i in t2_idx)
    
    # If one side is significantly stronger, assign it.
    # Reduced multiplier from 2.0 to 1.2 to be more sensitive to white jerseys.
    if t1_sum > 1.2 * t2_sum and t1_sum > 30:
        return 'team1'
    if t2_sum > 1.2 * t1_sum and t2_sum > 30:
        return 'team2'
    
    return 'unknown'

# 2. Track Loop
results = model.track(source="test_video.mp4", stream=True, tracker="bytetrack.yaml", conf=0.45)

first_frame_processed = False

for result in results:
    frame = result.orig_img
    if result.boxes.id is None:
        cv2.imshow("Lacrosse AI", frame)
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.int().cpu().tolist()

    # Calibrate as soon as we see enough players
    if not first_frame_processed and len(boxes) >= 4:
        basic_colors, t1_idx_list, t2_idx_list = auto_calibrate_teams(frame, boxes)
        if basic_colors is not None:
            first_frame_processed = True

    for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)
        
        if first_frame_processed and track_id not in player_teams:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                if track_id not in track_hist_data: track_hist_data[track_id] = []
                track_hist_data[track_id].append(create_color_histogram_nearest(crop, basic_colors))
                
                # Decision after 10 frames (faster than 20)
                if len(track_hist_data[track_id]) == 10:
                    avg_hist = np.mean(track_hist_data[track_id], axis=0)
                    player_teams[track_id] = assign_team(avg_hist, t1_idx_list, t2_idx_list)
                    del track_hist_data[track_id]

        # Visuals
        team = player_teams.get(track_id, "identifying...")
        
        # Team 1 = Yellow, Team 2 = White/Magenta, Unknown = Cyan
        color = (0, 255, 255) if team == 'team1' else (255, 255, 255) if team == 'team2' else (255, 255, 0)
        
        # Reduced the corner filter to be less aggressive
        if team != 'unknown' or (x1 > 20 and y1 > 20):
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id} {team}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Lacrosse AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.destroyAllWindows()


