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

## Data Visualization
![Adobe Express - LacrosseDemo](https://github.com/user-attachments/assets/579a6c58-875f-4e5b-a4fe-0d83c8b89547)

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

## Installation & Usage
### 1. Prerequisites
Before running the project, ensure you have the following installed:
* Python 3.10+
* pip
* A Roboflow API Key (required if you wish to re-run the training notebooks in the notebooks/ directory)

### 2. Environment Setup
Clone the repository and install the necessary computer vision and machine learning libraries. Run these commands in your terminal:
<img width="481" height="147" alt="Screenshot 2026-01-05 at 10 47 38 PM" src="https://github.com/user-attachments/assets/708477bd-7a43-41d4-b9b0-4a313aae7e5f" />

### 3. Running the Tracking System
Run in your terminal:
<img width="141" height="28" alt="Screenshot 2026-01-05 at 10 49 26 PM" src="https://github.com/user-attachments/assets/80a37fd1-8a63-4a49-96bc-6b10bf2a74f4" />
