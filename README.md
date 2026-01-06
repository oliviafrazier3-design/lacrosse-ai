# Lacrosse AI: Player Tracking & Team Analytics
## Project Overview
Designed an automated sports analytics pipeline to modernize the recruiting and scouting process for lacrosse coaches. By extracting metrics: player velocity, team affiliation, and performance stats, this system transforms raw game film into data. The goal is to provide scouts with a high-speed tool to identify top talent and streamline the manual video tagging process, allowing for more efficient and data driven recruiting decisions. 

## Key Features
* Multi-Object Tracking: Integrated YOLOv8 with ByteTrack to ensure consistent player ID persistence within a fast moving game
* Automated Team Classification: Developed custom identification logic using K-Means Clustering to group players based on jersey color signatures
* Lighting: Utilized HSV color space analysis to separate quality of color from brightness, ensuring accuracy despite shadows or sunlight 

## Technical Stack
* Language: Python
* Frameworks: YOLOv8, OpenCV
* Machine Learning: K-Means Clustering
* Data Management: Roboflow

## Challenges & Solutions
### 1. Environmental Noise Interference
Challenge: Background elements like field lines and endzones caused "White Team" players to be misidentified as "Yellow/Red Team"
Solution: Implemented 70% width torso crop to prioritize jersey pixels over background noise
* Developed a 30-frame temporal voting buffer; the system requires a statistical majority consensus across one second of footage before a team ID is locked, successfully eliminating "flickering" classifications

### 2. Variable Lighting Conditions
Challenge: Standard BGR color analysis failed when players moved from direct sunlight to shadow 
Solution: Now using HSV color space, allowing the system to focus on hue (color) while ignoring value (brightness) differences

## Data Visualization (need to do)
![lacrosse_ai_demo](https://github.com/user-attachments/assets/306278d5-20b1-4013-9e33-e020f36493ab)
![lacrosse_ai_demo](https://github.com/user-attachments/assets/306278d5-20b1-4013-9e33-e020f36493ab)


## Planned Additions
### 1. Player Identification & Analytics
* Jersey Number Recognition: Implementing OCR to map system Track IDs to official roster numbers
* Speed Mapping: Tracking the bottom center of players bounding boxes to calculate top-end and sustained sprinting speeds
* Automated Video Indexing: Creating a "searchable game" feature to generate video segments of specific players for recruting highlights

### 2. Possession & Ball Logic
* "Ball in Stick" Detection: Training a custom class to identify the stick head and ball proximity
* State-Based Possession Tracking: Implementing logic where the ball is assigned to a player's possession based on proximity and last known contact
* Even Classification: Detecting specific game states: ground balls, release points (shots/passes), cradle peaks

### 3. Tactical Intellegence 
* Spacial Heatmaps: Generating density maps to visualize field positioning, assessing defensive "in-position" and offensive spacing
* Automated Game Tagging: Mirroring Hudl style tagging to automatically break down game film into search able clips (goals, turnovers, clears)
* Shot Tracking: Utilizing goal-frame detection to automatically record and timestamp shots on goal

## Installation & Usage (need to do)

