# Lacrosse AI: Player Tracking & Team Analytics
## Project Overview
Designed an automated sports analytics pipeline to modernize the recruiting and scouting process for lacrosse coaches. By extracting metrics: player velocity, team affiliation, and performance stats, this system transforms raw game film into data. The goal is to provide scouts with a high-speed tool to identify top talent and streamline the manual video tagging process, allowing for more efficent and data driven recruiting decisions. 

## Key Features
* Multi-Object Tracking: Integrated YOLOv8 with ByteTrack to ensure consistent player ID persistance within a fast moving game
* Automated Team Classification: Developed custom identification logic using K-Means Clustering to group players based on jersey color signatures
* Lighting: Utilized HSV color space analysis to separate quality of color from brightness, ensuring accuracy despite shadows or sunlight 

## Techniacal Stack
* Language: Python
* Frameworks: YOLOv8, OpenCV
* Machine Learning: K-Means Clustering
* Data Management: Roboflow

## Challenges & Solutions
### 1. Enviornmental Noise Interference
Challenge: Background elements like field lines and endzones caused "White Team" players to be misidentified as "Yellow/Red Team"
Solution: Implemented 70% width torso crop to prioritize jersey pixels over background noise

### 2. Variable Lighting Conditions
Challenge: Standard BGR color analysis failed when players moved from direct sunlight to shadow 
Solution: Now using HSV color space, allowing the system to focus on hue (color) while ignoring value (brightness) differences

## Data Visualization

## Installation & Usage

