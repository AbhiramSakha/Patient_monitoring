# camera.py
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from deepface import DeepFace

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0) # Use 0 for Webcam
        
        # AI Models
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # State Variables (Shared with Frontend)
        self.current_emotion = "Neutral"
        self.last_alert = "Normal"
        self.alert_timestamp = ""
        self.prev_nose_y = 0
        self.lock = threading.Lock() # Thread safety

        # Start emotion detection thread
        self.emotion_thread = threading.Thread(target=self.detect_emotion_loop, daemon=True)
        self.emotion_thread.start()

    def detect_emotion_loop(self):
        """Runs continuously in background to check emotion every 2 seconds"""
        while True:
            if self.video.isOpened():
                success, frame = self.video.read()
                if success:
                    try:
                        # Analyze emotion on a small frame copy to save speed
                        objs = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                        with self.lock:
                            self.current_emotion = objs[0]['dominant_emotion']
                    except:
                        pass
            time.sleep(2)

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None, None

        # 1. Privacy Masking (Black Background)
        privacy_frame = np.zeros(frame.shape, dtype=np.uint8)
        
        # 2. Pose Detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        alert_status = "Normal"

        if results.pose_landmarks:
            # Draw Skeleton
            self.mp_drawing.draw_landmarks(
                privacy_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            # 3. Fall Detection Logic (Nose Drop)
            nose_y = results.pose_landmarks.landmark[0].y
            if self.prev_nose_y != 0 and (nose_y - self.prev_nose_y) > 0.04: # Threshold
                alert_status = "FALL DETECTED"
                with self.lock:
                    self.last_alert = "FALL DETECTED ⚠️"
                    self.alert_timestamp = time.strftime("%H:%M:%S")
            
            self.prev_nose_y = nose_y

        # Encode frame to JPEG for web streaming
        ret, jpeg = cv2.imencode('.jpg', privacy_frame)
        return jpeg.tobytes(), alert_status