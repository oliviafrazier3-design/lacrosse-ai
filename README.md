# Lacrosse AI: Player Tracking & Team Analytics
## Project Overview
The goal of this system is to automate the collection of sports performance data by tracking players and identifying team affiliation. Designed to streamline lacrosse film analysis, the pipeline will extract metrics such as player velocity, shots on goal 

## Key Features
* Multi-Object Tracking: Integrated YOLOv8 with ByteTrack to ensure consistent player ID persistance within a fast moving game
* Automated Team Classification: Developed custom identification logic using K-Means Clustering to group players based on jersey color signatures
* Lighting: Utilized HSV color-space analysis to separate quality of color from brightness, ensuring accuracy despite shadows or sunlight 

## Techniacal Stack
* Language: Python
* Frameworks: YOLOv8, OpenCV
* Machine Learning: K-Means Clustering
* Data Management: Roboflow

## Challenges & Solutions

