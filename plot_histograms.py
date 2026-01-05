import numpy as np
import matplotlib.pyplot as plt
import math

# 1. Load the data
data = np.load("player_histograms.npy", allow_pickle=True).item()
player_ids = list(data.keys())
num_players = len(player_ids)

if num_players == 0:
    print("No player data found.")
else:
    # 2. Grid Setup
    cols = 4  # Increased to 4 columns to make it wider/shorter
    rows = math.ceil(num_players / cols)

    # We set a smaller figsize and use 'constrained_layout' for better fitting
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2), constrained_layout=True)
    fig.suptitle("All Player Color Signatures", fontsize=10, fontweight='bold')

    axes_flat = axes.flatten()

    for i, p_id in enumerate(player_ids):
        ax = axes_flat[i]
        histogram = data[p_id]
        
        ax.bar(range(len(histogram)), histogram, color='blue', width=1.0)
        
        # --- MAKE THINGS SMALLER ---
        ax.set_title(f"ID: {p_id}", fontsize=9, pad=2)
        ax.tick_params(axis='both', which='major', labelsize=7) # Shrink axis numbers
        ax.grid(axis='y', alpha=0.2)

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    print(f"Displaying {num_players} players in a compact grid.")
    plt.show()
