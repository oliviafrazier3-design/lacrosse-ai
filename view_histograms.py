import numpy as np

# Step 1: Point to the file you saved earlier
file_name = "player_histograms.npy"

# Step 2: Load the data
# We use .item() at the end because we saved the whole thing as a dictionary object
data = np.load(file_name, allow_pickle=True).item()

print("="*40)
print(f"LOADED DATA: Found {len(data)} unique players")
print("="*40)

# Step 3: Loop through each player and look at their "Color Signature"
for track_id, histogram in data.items():
    print(f"\nPLAYER ID: {track_id}")
    
    # We only print the first 15 numbers so your screen isn't flooded
    # These numbers represent how many pixels of a certain color were found
    print(f"Color Signature (Sample): {histogram[:15]}")
    
    # Let's see the "Peak Color" - which bin has the most pixels?
    peak_bin = np.argmax(histogram)
    print(f"Dominant Color Bin: {peak_bin}")
    print("-" * 20)