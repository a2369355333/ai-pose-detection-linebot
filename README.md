# AI Pose Detection LINE Bot

A real-time pose detection system integrated with LINE messaging platform. The system detects hand gestures (left hand raised, right hand raised, or both hands raised) using MediaPipe and sends notifications through LINE Bot.

## 📋 Description

This project uses MediaPipe Pose detection to monitor hand positions in real-time via webcam. When specific hand gestures are detected, the system automatically sends notifications to LINE, making it perfect for fitness tracking, gesture control applications, or attendance systems.

## ✨ Features

- 🤚 Real-time hand gesture detection
  - Left hand raised detection
  - Right hand raised detection
  - Both hands raised detection
- 📱 LINE Bot integration for instant notifications
- 📊 Live FPS display
- 🎯 Pose landmark visualization
- 📝 Detection history logging
- ⏱️ Timestamp tracking for each action

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- LINE Bot account and channel access token
- pip (Python package installer)

### Installation

1. Install required dependencies
```bash
pip install -r requirements.txt
```

2. Set up LINE Bot
   - Create a LINE Bot account at [LINE Developers Console](https://developers.line.biz/)
   - Get your Channel Access Token
   - Get your User ID or Group ID

3. Configure LINE credentials
   - Open `line_bot.py`
   - Replace the following with your credentials:
```python
   LINE_CHANNEL_ACCESS_TOKEN = 'your_channel_access_token_here'
   LINE_USER_ID = 'your_user_id_here'
```

### Getting Started

1. Run the pose detection script
```bash
python line_bot.py
```

2. The application will:
   - Initialize the webcam
   - Start detecting pose landmarks
   - Monitor hand positions
   - Send LINE notifications when gestures are detected
   - Display real-time video feed with pose overlay

3. Perform gestures:
   - Raise your left hand above shoulder level
   - Raise your right hand above shoulder level
   - Raise both hands above shoulder level

4. Press `q` to quit the application


### Adjust Detection Sensitivity

In `line_bot.py`, modify these parameters:
```python
# Confidence threshold for pose detection
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

# Cooldown period between notifications (seconds)
notification_cooldown = 3
```

### Customize LINE Messages

Edit the notification messages in the detection logic:
```python
message = "Left Hand Raised detected!"
message = "Right Hand Raised detected!"
message = "Both Hands Raised detected!"
```

### Demo
<img width="895" height="508" alt="image" src="https://github.com/user-attachments/assets/84321380-d163-45a7-9a7d-ce04444bea33" />
