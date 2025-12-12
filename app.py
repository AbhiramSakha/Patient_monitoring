# ============================
# MediGuard ICU – FINAL UPDATED (1 Webcam Mode) with improved controls
# ============================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import signal
import sys
from deepface import DeepFace
from gtts import gTTS
from flask import Flask, render_template, Response, jsonify, request
from werkzeug.utils import secure_filename

# ------------------ CONFIG ------------------
PATIENT_COUNT = 16
UPLOAD_FOLDER = "uploads"
VOICE_FOLDER = "static/voice"
CLIP_FOLDER = "clips"

for d in (UPLOAD_FOLDER, VOICE_FOLDER, CLIP_FOLDER):
    os.makedirs(d, exist_ok=True)

FPS_FALLBACK = 24
DEEPFACE_INTERVAL = 1.2   # emotion every 1.2 sec

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------------ MODEL FLAGS ------------------
DEEPFACE_READY = False
POSE_READY = False
mp_pose = None

# ------------------ BACKGROUND MODEL LOADING ------------------
def load_models_async():
    global mp_pose, POSE_READY, DEEPFACE_READY
    print("Loading MediaPipe Pose in background...")
    try:
        mp_pose = mp.solutions.pose.Pose(model_complexity=0)
        POSE_READY = True
        print("MediaPipe Pose loaded.")
    except Exception as e:
        POSE_READY = False
        print("MediaPipe Pose failed to load:", e)

    print("Warming DeepFace in background (silent)...")
    try:
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        DeepFace.analyze(dummy, actions=["emotion"], enforce_detection=False, silent=True)
        DEEPFACE_READY = True
        print("DeepFace warm-up OK.")
    except Exception as e:
        DEEPFACE_READY = False
        print("DeepFace warm-up failed:", e)

threading.Thread(target=load_models_async, daemon=True).start()

# ------------------ TTS GENERATION ------------------
def generate_tts_async():
    alerts = {
        "FALL": "Fall detected! Immediate help needed.",
        "PAIN": "The patient is in pain! Please check immediately.",
        "PANIC": "Panic detected! Urgent attention needed.",
        "SAFE": "Patient is stable.",
        "FEVERISH": "Patient may be feverish.",
        "WEAK": "Patient appears weak.",
        "EXHAUSTED": "Patient looks exhausted."
    }
    for name, text in alerts.items():
        path = os.path.join(VOICE_FOLDER, f"{name.lower()}.mp3")
        if not os.path.exists(path):
            try:
                g = gTTS(text=text, lang="en")
                g.save(path)
                print("Generated TTS:", name)
            except Exception as e:
                print("TTS generation failed for", name, e)

threading.Thread(target=generate_tts_async, daemon=True).start()

# ------------------ ONE CAMERA SHARED ------------------
print("Initializing main webcam...")
MAIN_CAM = cv2.VideoCapture(0)
if MAIN_CAM.isOpened():
    MAIN_CAM.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    MAIN_CAM.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Main webcam opened.")
else:
    print("ERROR: Webcam not detected! Using placeholders until available.")

# ------------------ PatientMonitor ------------------
class PatientMonitor:
    def __init__(self, pid, name):
        self.pid = pid
        self.name = name
        self.frame = self.placeholder()
        self.emotion = "neutral"
        self.posture = "Unknown"
        self.cry = 0.0
        self.panic = 0.0
        self.motion_score = 0.0
        self.alert = "SAFE"

        self.mode = "camera"   # camera / video / paused
        self.video_path = None
        self.video_cap = None
        self.video_finished = False

        self.last_emo_time = 0.0
        self.prev_gray = None
        self.motion_window = []

        self.clip_buffer = []
        self.clip_lock = threading.Lock()

        # Start update & analyze threads
        threading.Thread(target=self.update_loop, daemon=True).start()
        threading.Thread(target=self.analyze_loop, daemon=True).start()

    def placeholder(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"Initializing Camera...", (70, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
        return img

    def _read_from_main_cam(self):
        """Read frame from MAIN_CAM (shared webcam)."""
        global MAIN_CAM
        if MAIN_CAM and MAIN_CAM.isOpened():
            ok, frame = MAIN_CAM.read()
            if ok:
                return frame
        return None

    def _read_from_video(self):
        """Read from assigned video capture if in video mode."""
        if self.video_cap:
            ok, frame = self.video_cap.read()
            if ok:
                return frame
            else:
                # video ended or failed
                try:
                    self.video_cap.release()
                except:
                    pass
                self.video_cap = None
                self.video_finished = True
                self.mode = "camera"
        return None

    # ------------ LIVE CAMERA FRAME LOOP ------------
    def update_loop(self):
        while True:
            frame = None
            if self.mode == "camera":
                frame = self._read_from_main_cam()
            elif self.mode == "video":
                frame = self._read_from_video()
            elif self.mode == "paused":
                # just keep last frame
                frame = self.frame

            if frame is not None:
                self.frame = frame

                # Motion calculation (cheap)
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if self.prev_gray is None:
                        self.prev_gray = gray
                    diff = cv2.absdiff(self.prev_gray, gray)
                    motion = float(np.mean(diff))
                    self.prev_gray = gray

                    self.motion_window.append(motion)
                    if len(self.motion_window) > 6:
                        self.motion_window.pop(0)
                    self.motion_score = float(np.mean(self.motion_window))
                except Exception:
                    self.motion_score = 0.0

                # push to clip buffer for emergency saving
                with self.clip_lock:
                    self.clip_buffer.append(frame.copy())
                    max_frames = int(FPS_FALLBACK * 3)  # 3 second clip
                    if len(self.clip_buffer) > max_frames:
                        self.clip_buffer.pop(0)

            time.sleep(0.02)  # gentle throttle

    # ------------ AI ANALYSIS LOOP ------------
    def analyze_loop(self):
        global DEEPFACE_READY, POSE_READY, mp_pose
        while True:
            now = time.time()
            if now - self.last_emo_time < DEEPFACE_INTERVAL:
                time.sleep(0.05)
                continue

            frame = self.frame
            if frame is None:
                time.sleep(0.05)
                continue

            # DeepFace analysis (non-blocking style)
            try:
                df = DeepFace.analyze(frame, actions=["emotion"],
                                      enforce_detection=False, silent=True)
                emo = df[0].get("dominant_emotion", self.emotion)
                em_map = df[0].get("emotion", {})
                self.emotion = emo
                self.cry = float(em_map.get("sad", 0) + em_map.get("fear", 0))
                self.panic = float(em_map.get("fear", 0))
            except Exception:
                # keep last known
                pass

            # Mediapipe Pose – wrap in safe try/except to avoid timestamp mismatches
            try:
                if POSE_READY and mp_pose is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = mp_pose.process(rgb)  # safe usage
                    if res and res.pose_landmarks:
                        lm = res.pose_landmarks.landmark
                        # check shoulders index 11 & 12
                        if len(lm) > 12:
                            diff = abs(lm[11].x - lm[12].x)
                            self.posture = "LYING" if diff < 0.05 else "SITTING"
                        else:
                            self.posture = "Unknown"
                else:
                    self.posture = self.posture  # keep current
            except Exception:
                # swallow errors from MediaPipe internals
                pass

            # FEVERISH detection (simple color heuristic)
            try:
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 3
                y1 = max(0, cy - 40); y2 = min(h, cy + 40)
                x1 = max(0, cx - 40); x2 = min(w, cx + 40)
                face_roi = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else frame
                b_avg, g_avg, r_avg = cv2.mean(face_roi)[:3]
                brightness = float(np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))) if face_roi.size else 0.0
                feverish = (r_avg > 140 and r_avg > (g_avg + 10) and brightness > 50)
            except Exception:
                feverish = False

            # ALERT LOGIC (priority)
            new_alert = "SAFE"
            if self.cry > 45:
                new_alert = "PAIN"
            elif self.panic > 55:
                new_alert = "PANIC"
            elif self.posture == "LYING":
                new_alert = "FALL"
            elif feverish:
                new_alert = "FEVERISH"
            elif self.motion_score < 3.0:
                new_alert = "WEAK"
            else:
                new_alert = "SAFE"

            self.alert = new_alert
            self.last_emo_time = time.time()

            time.sleep(0.05)

    # ------------ STREAM (multipart jpeg) ------------
    def stream(self):
        while True:
            if self.frame is None:
                # placeholder image
                img = self.placeholder()
            else:
                img = self.frame.copy()

            cv2.putText(img, self.name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            try:
                _, jpeg = cv2.imencode(".jpg", img)
                yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" +
                       jpeg.tobytes() + b"\r\n")
            except Exception:
                # in case encoding fails, wait and continue
                time.sleep(0.05)
                continue

            time.sleep(0.015)  # small delay to prevent too tight loop

    # ------------ VITALS API ------------
    def vitals(self):
        return {
            "id": self.pid,
            "name": self.name,
            "emotion": self.emotion,
            "posture": self.posture,
            "cry": round(self.cry, 2),
            "panic": round(self.panic, 2),
            "motion": round(self.motion_score, 2),
            "alert": self.alert,
            "mode": self.mode,
            "video_path": self.video_path
        }

    # ------------ Video / Camera Controls ------------
    def load_video(self, path):
        """Load a video file and switch to video playback."""
        try:
            if self.video_cap:
                try:
                    self.video_cap.release()
                except:
                    pass
                self.video_cap = None

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"[upload] video for pid {self.pid} failed to open: {path}")
                return False
            self.video_cap = cap
            self.video_path = path
            self.mode = "video"
            self.video_finished = False
            print(f"[upload] patient {self.pid} now playing video {path}")
            return True
        except Exception as e:
            print("load_video error:", e)
            return False

    def switch_to_camera(self):
        """Switch back to shared camera."""
        self.mode = "camera"
        self.video_path = None
        if self.video_cap:
            try:
                self.video_cap.release()
            except:
                pass
            self.video_cap = None

    def pause(self):
        self.mode = "paused"

    def resume(self):
        # if there is a video path, resume video; else camera
        if self.video_path:
            self.mode = "video"
        else:
            self.mode = "camera"

    def save_emergency_clip(self, seconds=3):
        with self.clip_lock:
            frames = list(self.clip_buffer)
        if not frames:
            return None
        height, width = frames[0].shape[:2]
        fname = f"clip_pid{self.pid}_{int(time.time())}.mp4"
        fpath = os.path.join(CLIP_FOLDER, fname)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(fpath, fourcc, FPS_FALLBACK, (width, height))
        for f in frames:
            out.write(f)
        out.release()
        print(f"Saved emergency clip: {fpath}")
        return fpath

# ------------------ CREATE 16 PATIENTS ------------------
patients = [PatientMonitor(i, f"Patient {i}") for i in range(1, PATIENT_COUNT + 1)]

# ------------------ ROUTES ------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/patient/<int:pid>")
def patient_view(pid):
    if pid < 1 or pid > len(patients):
        return "Invalid patient id", 404
    return render_template("patient_view.html", patient_id=pid)

@app.route("/video/<int:pid>")
def video(pid):
    if pid < 1 or pid > len(patients):
        return "Invalid patient id", 404
    return Response(
        patients[pid-1].stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/vitals/<int:pid>")
def get_vitals(pid):
    if pid < 1 or pid > len(patients):
        return jsonify({"error":"invalid pid"}), 404
    return jsonify(patients[pid-1].vitals())

@app.route("/api/dashboard")
def dashboard_api():
    return jsonify([p.vitals() for p in patients])

@app.route("/upload/<int:pid>", methods=["POST"])
def upload(pid):
    if pid < 1 or pid > len(patients):
        return jsonify({"success": False, "error": "invalid pid"})
    f = request.files.get("video")
    if not f:
        return jsonify({"success": False, "error": "no file"})
    filename = secure_filename(f.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(path)
    ok = patients[pid-1].load_video(path)
    return jsonify({"success": ok})

@app.route("/switch_camera/<int:pid>", methods=["POST"])
def switch_camera_route(pid):
    if pid < 1 or pid > len(patients):
        return jsonify({"success": False, "error": "invalid pid"})
    patients[pid-1].switch_to_camera()
    return jsonify({"success": True})

@app.route("/pause/<int:pid>", methods=["POST"])
def pause(pid):
    if pid < 1 or pid > len(patients):
        return jsonify({"success": False, "error": "invalid pid"})
    patients[pid-1].pause()
    return jsonify({"success": True})

@app.route("/resume/<int:pid>", methods=["POST"])
def resume(pid):
    if pid < 1 or pid > len(patients):
        return jsonify({"success": False, "error": "invalid pid"})
    patients[pid-1].resume()
    return jsonify({"success": True})

@app.route("/save_clip/<int:pid>", methods=["POST"])
def save_clip(pid):
    if pid < 1 or pid > len(patients):
        return jsonify({"success": False, "error": "invalid pid"})
    fpath = patients[pid-1].save_emergency_clip()
    if fpath:
        return jsonify({"success": True, "path": fpath})
    return jsonify({"success": False, "error": "no clip"})

# ------------------ Graceful shutdown ------------------
def _cleanup_and_exit(signum=None, frame=None):
    print("Shutting down — releasing camera(s)...")
    try:
        if MAIN_CAM:
            MAIN_CAM.release()
    except:
        pass
    try:
        for p in patients:
            if getattr(p, "video_cap", None):
                try:
                    p.video_cap.release()
                except:
                    pass
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, _cleanup_and_exit)
signal.signal(signal.SIGTERM, _cleanup_and_exit)

# ------------------ RUN ------------------
if __name__ == "__main__":
    print("MediGuard ICU – FINAL VERSION (updated) STARTING...")
    print("Open http://127.0.0.1:5000/dashboard")
    app.run(host="0.0.0.0", port=5000, debug=False)
