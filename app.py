import os
import webbrowser
from threading import Timer
os.environ["FLASK_SOCKETIO_ASYNC_MODE"] = "threading"
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import sys
from unittest.mock import MagicMock
# Prevent mediapipe from pulling in the full tensorflow library
mock_tf = MagicMock()
sys.modules['tensorflow'] = mock_tf
sys.modules['tensorflow.tools'] = MagicMock()
sys.modules['tensorflow.tools.docs'] = MagicMock()

import cv2
import numpy as np
import mediapipe as mp

# Resilient MediaPipe solutions import for PyInstaller
try:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as mp_draw
except ImportError:
    try:
        import mediapipe.python.solutions.hands as mp_hands
        import mediapipe.python.solutions.drawing_utils as mp_draw
    except ImportError:
        # Fallback to direct attribute access if imports fail
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils

import joblib
from collections import deque, Counter

import uuid
import sys
import time
import threading
from deep_translator import GoogleTranslator

app = Flask(__name__)
socketio = SocketIO(
    app,
    async_mode="threading",
    cors_allowed_origins="*"
)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ==============================
# ML MODELS & LOGIC
# ==============================
class SignLanguageSystem:
    def __init__(self):
        # Load Models
        print("Loading models...")
        self.static_model = joblib.load(resource_path("isl_alphabet_model.pkl"), mmap_mode=None)
        self.dynamic_model = None
        
        self.STATIC_LABELS = [chr(i) for i in range(65, 91)]
        self.DYNAMIC_LABELS = ["HELLO", "THANK YOU"]
        
        # Mediapipe
        self.mp_hands = mp_hands
        self.mp_draw = mp_draw
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Constants
        self.MOTION_THRESHOLD = 0.1
        self.STATIC_FRAMES = 5
        self.DYNAMIC_FRAMES = 30
        
        # State
        self.prev_keypoints = None
        self.stable_count = 0
        self.static_locked = False
        self.dynamic_sequence = deque(maxlen=self.DYNAMIC_FRAMES)
        
        self.sentence = ""
        self.display_sign = ""
        self.can_add_space = False
        self.language = "en"
        self.target_sentence = ""
        
        self.camera = cv2.VideoCapture(0)

    def smart_refine(self, text):
        """Simple refinement to make detections more readable"""
        if not text: return ""
        # Remove multiple spaces, capitalize first letter, add period
        refined = " ".join(text.split())
        if refined:
            refined = refined.capitalize()
            if not refined.endswith("."):
                refined += "."
        return refined

    def translate_sentence(self):
        """Translates the sentence based on chosen language"""
        refined_en = self.smart_refine(self.sentence)
        if self.language == "hi":
            try:
                self.target_sentence = GoogleTranslator(source='auto', target='hindi').translate(refined_en)
            except:
                self.target_sentence = refined_en + " (Translation Error)"
        else:
            self.target_sentence = refined_en

    def extract_keypoints(self, results):
        left = np.zeros(63)
        right = np.zeros(63)
        if results.multi_hand_landmarks and results.multi_handedness:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[i].classification[0].label
                points = []
                for lm in hand_landmarks.landmark:
                    points.extend([lm.x, lm.y, lm.z])
                if label == "Left":
                    left = np.array(points)
                else:
                    right = np.array(points)
        return np.concatenate([left, right])

    def process_frame(self):
        success, frame = self.camera.read()
        if not success:
            return None

        frame = cv2.flip(frame, 1) # Flip to reverse mirroring if source is mirrored
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        # Draw
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hl, self.mp_hands.HAND_CONNECTIONS)
        
        # Logic
        if not results.multi_hand_landmarks:
            if self.can_add_space:
                if self.sentence and not self.sentence.endswith(" "):
                    self.update_sentence(" ")
                self.can_add_space = False
            self.prev_keypoints = None
            self.stable_count = 0
            self.static_locked = False
            self.display_sign = ""
        else:
            keypoints = self.extract_keypoints(results)
            motion = 0
            if self.prev_keypoints is not None:
                motion = np.linalg.norm(keypoints - self.prev_keypoints)
            self.prev_keypoints = keypoints

            # Debug Motion
            # print(f"Motion: {motion:.4f} | Stable: {self.stable_count}")

            # Dynamic
            if motion > self.MOTION_THRESHOLD:
                self.stable_count = 0
                self.static_locked = False
                self.dynamic_sequence.append(keypoints)
                
                if len(self.dynamic_sequence) == self.DYNAMIC_FRAMES:
                    # Lazy Load ONNX Model
                    if self.dynamic_model is None:
                        print("Lazy loading ONNX model...")
                        import onnxruntime as ort
                        self.dynamic_model = ort.InferenceSession(resource_path("dynamic_sign_model.onnx"))
                        self.input_name = self.dynamic_model.get_inputs()[0].name
                    
                    X = np.array(self.dynamic_sequence, dtype=np.float32).reshape(1, self.DYNAMIC_FRAMES, 126)
                    try:
                        pred = self.dynamic_model.run(None, {self.input_name: X})[0][0]
                        
                        confidence = np.max(pred)
                        label_idx = np.argmax(pred)
                        
                        if confidence > 0.75:
                            self.display_sign = self.DYNAMIC_LABELS[label_idx]
                            self.update_sentence(self.display_sign + " ")
                            self.can_add_space = True
                            self.dynamic_sequence.clear()
                    except:
                        pass
            
            # Static
            else:
                self.stable_count += 1
                if self.stable_count >= self.STATIC_FRAMES and not self.static_locked:
                    X = keypoints.reshape(1, -1)
                    try:
                        pred = self.static_model.predict(X)[0]
                        # print(f"Static Pred: {pred}")
                        if 0 <= pred < len(self.STATIC_LABELS):
                            self.display_sign = self.STATIC_LABELS[pred]
                            self.update_sentence(self.display_sign)
                            self.can_add_space = True
                            self.static_locked = True
                            self.dynamic_sequence.clear()
                    except Exception as e:
                        print(f"Static Error: {e}")
                        pass
                # self.stable_count = 0  <-- REMOVED incorrect reset logic


        # Send updates to frontend via SocketIO
        socketio.emit('update_status', {
            'sign': self.display_sign, 
            'sentence': self.target_sentence or self.sentence # Fallback to raw if logic is pending
        })
        
        return frame

    def update_sentence(self, new_text):
        self.sentence += new_text
        self.translate_sentence() # Update translation immediately

    def clear(self):
        self.sentence = ""
        self.target_sentence = ""
    
    def backspace(self):
        if not self.sentence: return
        if self.sentence.endswith(" "):
            self.sentence = self.sentence[:-1]
        parts = self.sentence.rstrip().split(" ")
        if len(parts) > 1:
            self.sentence = " ".join(parts[:-1]) + " "
        else:
            self.sentence = self.sentence[:-1]
        self.translate_sentence() # Update translation immediately

# Global Instance
system = SignLanguageSystem()

# ==============================
# ROUTES
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        frame = system.process_frame()
        if frame is None:
            break
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Add slight delay to reduce CPU usage and release GIL
        time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==============================
# SOCKET EVENTS
# ==============================
@socketio.on('command')
def handle_command(data):
    cmd = data.get('action')
    if cmd == 'clear':
        system.clear()
    elif cmd == 'backspace':
        system.backspace()
    elif cmd == 'speak':
        # Hook up TTS here if needed, or do it frontend side
        pass

@socketio.on('set_language')
def handle_set_language(data):
    system.language = data.get('language', 'en')
    system.translate_sentence()
    # Force update
    socketio.emit('update_status', {'sign': system.display_sign, 'sentence': system.target_sentence})

@socketio.on('translate_now')
def handle_translate_now(data):
    text = data.get('text')
    target_lang = data.get('target', 'hi')
    if text:
        try:
            translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
            socketio.emit('stt_translation', {'translated': translated})
        except Exception as e:
            print(f"Translation Error: {e}")
            socketio.emit('stt_translation', {'translated': text + " (Translation Error)"})

def open_browser():
    """Automatically open the browser to the app's URL."""
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    # Automatically open browser after 1.5 seconds
    Timer(1.5, open_browser).start()
    
    # Disable debug mode to prevent memory-mapped debugger overhead which causes "Paging file too small" errors
    socketio.run(app, debug=False, port=5000, allow_unsafe_werkzeug=True)
