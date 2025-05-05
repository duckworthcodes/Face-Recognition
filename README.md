# School Security System with AI-Powered Motion Detection

## Overview
This project implements an intelligent security system designed for educational institutions, combining computer vision and machine learning to provide real-time monitoring and threat detection. The system uses the MobileNet SSD model for person detection and implements motion detection algorithms to identify suspicious activities.

## Features
- Real-time video monitoring with motion detection
- AI-powered person detection using MobileNet SSD
- Configurable alarm system with visual and audio alerts
- Multi-threaded architecture for efficient processing
- User-friendly interface with keyboard controls

## Technical Stack
- Python 3.x
- OpenCV (cv2)
- MobileNet SSD (Deep Learning Model)
- NumPy
- Threading for concurrent operations

## Prerequisites
- Python 3.x installed
- Webcam or IP camera
- Required Python packages (install using pip):
  ```
  pip install opencv-python
  pip install numpy
  pip install imutils
  ```

## Installation
1. Clone this repository:
   ```
   git clone [repository-url]
   ```
2. Download the required model files:
   - `deploy.prototxt`
   - `MobileNetSSD_deploy.caffemodel`
3. Place the model files in the project directory

## Usage
1. Run the main script:
   ```
   python refresh.py
   ```
2. Controls:
   - Press 't' to toggle alarm mode
   - Press 'q' to quit the application

## Configuration
The system can be configured by modifying the following constants in `refresh.py`:
- `THRESHOLD_VALUE`: Motion detection sensitivity (default: 25)
- `MOTION_THRESHOLD`: Minimum motion required to trigger detection (default: 300)
- `ALARM_THRESHOLD`: Number of consecutive detections before alarm triggers (default: 20)

## Project Structure
```
├── refresh.py              # Main application file
├── deploy.prototxt         # MobileNet SSD model configuration
├── MobileNetSSD_deploy.caffemodel  # Pre-trained model weights
└── README.md              # Project documentation
```

## Features in Detail
1. **Motion Detection**
   - Real-time frame differencing
   - Gaussian blur for noise reduction
   - Configurable sensitivity thresholds

2. **Person Detection**
   - Uses MobileNet SSD for accurate person detection
   - Confidence threshold of 0.5 for reliable detections
   - Real-time bounding box visualization

3. **Alarm System**
   - Visual alerts through console logging
   - Audio alerts using system beeps
   - Configurable alarm triggers

## Contributing
Feel free to submit issues and enhancement requests!

## Author
[Vaibhav Verma] 

