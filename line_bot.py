import cv2
import time
import csv
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
from linebot import LineBotApi
from linebot.models import TextSendMessage

class PoseDetector:
    def __init__(self, camera_id=0, min_detection_confidence=0.5, min_tracking_confidence=0.5, 
                 csv_file="pose_records.csv", line_token=None, user_id=None):

        self.camera_id = camera_id
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.csv_file = csv_file
        self.cap = None

        # LINE Bot setup
        self.line_bot_api = LineBotApi(line_token) if line_token else None
        self.user_id = user_id
        self.line_available = (self.line_bot_api is not None and self.user_id is not None)

        # Initialize CSV file
        self._init_csv()

        # Last recorded action and time
        self.last_action = None
        self.last_action_time = 0
        self.cooldown_time = 3

        # Set landmark index
        self.LEFT_WRIST = self.mp_pose.PoseLandmark.LEFT_WRIST.value
        self.RIGHT_WRIST = self.mp_pose.PoseLandmark.RIGHT_WRIST.value
        self.LEFT_SHOULDER = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        self.RIGHT_SHOULDER = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        self.LEFT_ELBOW = self.mp_pose.PoseLandmark.LEFT_ELBOW.value
        self.RIGHT_ELBOW = self.mp_pose.PoseLandmark.RIGHT_ELBOW.value
        self.LEFT_HIP = self.mp_pose.PoseLandmark.LEFT_HIP.value
        self.RIGHT_HIP = self.mp_pose.PoseLandmark.RIGHT_HIP.value

        # 手勢檢測參數
        self.HAND_RAISE_THRESHOLD = 0.03   # 手舉起的基本閾值（非常敏感）
        self.CONFIDENCE_THRESHOLD = 0.3    # 關鍵點的置信度
        self.STABILITY_FRAMES = 2          # 連續檢測幀數
        self.gesture_history = []          # 儲存檢測結果
        
        # 防止雙手放下時誤觸發的冷卻機制
        self.last_both_hands_time = 0
        self.single_hand_cooldown = 1.0    # 雙手放下後1秒內不檢測單手

    def _init_csv(self):
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Action'])

    def _get_english_action_name(self, action):
        action_map = {
            "雙手舉起": "Both Hands Raised",
            "左手舉起": "Left Hand Raised",
            "右手舉起": "Right Hand Raised",
        }
        return action_map.get(action, action)

    def _send_to_line(self, action, timestamp):
        try:
            action_english = self._get_english_action_name(action)
            message = f"[{timestamp}] Detected: {action_english}"
            df = pd.read_csv(self.csv_file, encoding='utf-8')

            if action == '雙手舉起':
                df_filtered = df[df['Action'] == '雙手舉起']
                if not df_filtered.empty:
                    # Add summary to message
                    message += f'\n\nTotal {action_english} detections: {len(df_filtered)}'
                    # Add latest 3 records
                    message += f"\n\nLatest records:\n{df_filtered.tail(3).to_string(index=False)}"
            
            # Send LINE message
            self.line_bot_api.push_message(
                self.user_id, 
                TextSendMessage(text=message)
            )
            print(f"LINE notification sent: {action_english}") 
        except Exception as e:
            print(f"Error sending LINE notification: {str(e)}")

    def _record_action(self, action):
        current_time = time.time()

        # 只有當檢測到明確的手勢時才記錄
        if action is None:
            return

        if self.last_action == action and (current_time - self.last_action_time) < self.cooldown_time:
            return

        self.last_action = action
        self.last_action_time = current_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, action])

        action_english = self._get_english_action_name(action)
        print(f'Action recorded: {action_english}')

        if self.line_available:
            self._send_to_line(action, timestamp)

    def detect_gestures(self, landmarks):
        """
        簡化且敏感的手勢檢測邏輯
        """
        # 獲取關鍵點
        left_wrist = landmarks.landmark[self.LEFT_WRIST]
        right_wrist = landmarks.landmark[self.RIGHT_WRIST]
        left_shoulder = landmarks.landmark[self.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.RIGHT_SHOULDER]

        # 檢查關鍵點的可見性
        if (left_wrist.visibility < self.CONFIDENCE_THRESHOLD or 
            right_wrist.visibility < self.CONFIDENCE_THRESHOLD or
            left_shoulder.visibility < self.CONFIDENCE_THRESHOLD or 
            right_shoulder.visibility < self.CONFIDENCE_THRESHOLD):
            return None

        current_time = time.time()

        # 簡單直接的檢測邏輯：手腕高於肩膀就算舉起
        left_raised = left_wrist.y < (left_shoulder.y - self.HAND_RAISE_THRESHOLD)
        right_raised = right_wrist.y < (right_shoulder.y - self.HAND_RAISE_THRESHOLD)

        # 判斷手勢
        current_gesture = None
        
        if left_raised and right_raised:
            current_gesture = '雙手舉起'
            self.last_both_hands_time = current_time  # 記錄雙手舉起的時間
        elif left_raised and not right_raised:
            # 檢查是否在雙手放下的冷卻期間
            if current_time - self.last_both_hands_time > self.single_hand_cooldown:
                current_gesture = '左手舉起'
        elif not left_raised and right_raised:
            # 檢查是否在雙手放下的冷卻期間
            if current_time - self.last_both_hands_time > self.single_hand_cooldown:
                current_gesture = '右手舉起'

        # 使用歷史記錄穩定檢測結果
        self.gesture_history.append(current_gesture)
        if len(self.gesture_history) > self.STABILITY_FRAMES:
            self.gesture_history.pop(0)

        # 連續檢測到相同手勢才確認
        if len(self.gesture_history) == self.STABILITY_FRAMES:
            if all(gesture == current_gesture for gesture in self.gesture_history):
                return current_gesture

        return None

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2)
            )

            action = self.detect_gestures(results.pose_landmarks)

            if action:
                action_english = self._get_english_action_name(action)
                cv2.putText(frame, f'Action: {action_english}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self._record_action(action)
            else:
                # 顯示當前檢測狀態
                if len(self.gesture_history) > 0 and self.gesture_history[-1]:
                    current_gesture = self.gesture_history[-1]
                    action_english = self._get_english_action_name(current_gesture)
                    cv2.putText(frame, f'Detecting: {action_english}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Action: No Gesture', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'Action: No Pose Detected', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame

    def connect_camera(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise Exception('Cannot connect to camera')
            
            # Set camera resolution to 640x480
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            return True
        except Exception as e:
            print(f"Error connecting to camera: {str(e)}")
            return False

    def run(self):
        if not self.connect_camera():
            return
        
        pre_time = 0

        while True:
            try:
                success, frame = self.cap.read()
                if not success:
                    print('Cannot read image')
                    break

                # Calculate and display FPS    
                processed_frame = self.process_frame(frame)
                current_time = time.time()
                fps = 1 / (current_time - pre_time)
                pre_time = current_time

                cv2.putText(processed_frame, f'FPS: {int(fps)}', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show LINE status
                if self.line_available:
                    cv2.putText(processed_frame, f'LINE: Connected', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(processed_frame, f'LINE: Not configured', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Show image
                cv2.imshow("Pose Detection", processed_frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"Error during processing: {str(e)}")
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        """Destructor, ensure resources are properly released"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


def test_line_notification(line_token, user_id):
    try:
        line_bot_api = LineBotApi(line_token)
        message = f"[TEST] Hand gesture detection system is now connected ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
        line_bot_api.push_message(user_id, TextSendMessage(text=message))
        print("LINE test message sent successfully!")
        return True
    except Exception as e:
        print(f"Error testing LINE notification: {str(e)}")
        return False

if __name__ == '__main__':
    # Change your line token & user id
    LINE_TOKEN = '17sSAsWQ3/WdENbkvkdxFPf3TKd+kKs8XLbYGu/JeKA+jkUmzQwnQYkzXHUVOYEDnQrB2Y6G3RvT+Ox6QekDmtuHfpgYtTuYW6rerghIO5IPTLeDZ9d37BytWD5r7R5nu22TNrZKAc6gi1RWu5vwjwdB04t89/1O/w1cDnyilFU='
    USER_ID = 'U333dde09b20397a09ae5ecc2ea19d445' 
    line_configured = False

    if LINE_TOKEN != '' and USER_ID != '':
        print('Testing LINE notification....')
        line_configured = test_line_notification(LINE_TOKEN, USER_ID)
    
    detector = PoseDetector(
        camera_id=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        csv_file="pose_records.csv",
        line_token=LINE_TOKEN if line_configured else None,
        user_id=USER_ID if line_configured else None
    )

    detector.run()