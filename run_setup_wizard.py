import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import time
import subprocess
import sys

# Configurations
STATIC_SIGNS = [chr(i) for i in range(65, 91)] # A to Z
DYNAMIC_SIGNS = ["hello", "thank_you"]
DATASET_DIR = "dataset"
DYNAMIC_DIR = "dynamic_dataset"

# Direct access handles MediaPipe 0.10.x imports on some setups
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def wait_for_key(cap, message, key=' '):
    while True:
        ret, frame = cap.read()
        if not ret: return False
        frame = cv2.flip(frame, 1)
        # Put text with background for readability
        cv2.rectangle(frame, (0, 0), (640, 60), (0,0,0), -1)
        cv2.putText(frame, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("SignBridge Setup", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(key): return True
        if k == 27: return False # ESC

def collect_static(cap):
    os.makedirs(DATASET_DIR, exist_ok=True)
    for sign in STATIC_SIGNS:
        sign_path = os.path.join(DATASET_DIR, sign)
        os.makedirs(sign_path, exist_ok=True)
        csv_path = os.path.join(sign_path, "data.csv")
        
        if not wait_for_key(cap, f"Press SPACE to start recording '{sign}' (Static)"): 
            return False

        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = []
            for hand in ["L", "R"]:
                for i in range(21):
                    header.extend([f"{hand}_{i}_x", f"{hand}_{i}_y", f"{hand}_{i}_z"])
            writer.writerow(header)
            
            samples = 0
            while samples < 30: # 30 samples per letter
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                row = []
                
                if results.multi_hand_landmarks:
                    for hl in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                        for lm in hl.landmark: row.extend([lm.x, lm.y, lm.z])
                    # Pad to 126
                    while len(row) < 126: row.extend([0, 0, 0])
                    # Limit to 126 to prevent bugs
                    row = row[:126]
                    writer.writerow(row)
                    samples += 1
                    
                cv2.rectangle(frame, (0, 0), (640, 60), (0,0,0), -1)
                cv2.putText(frame, f"Recording '{sign}': {samples}/30", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("SignBridge Setup", frame)
                if cv2.waitKey(10) & 0xFF == 27: return False
    return True

def collect_dynamic(cap):
    os.makedirs(DYNAMIC_DIR, exist_ok=True)
    for sign in DYNAMIC_SIGNS:
        sign_path = os.path.join(DYNAMIC_DIR, sign)
        os.makedirs(sign_path, exist_ok=True)
        
        samples_needed = 20 # 20 sequences per word
        sequences_collected = 0
        
        while sequences_collected < samples_needed:
            if not wait_for_key(cap, f"Press SPACE to record '{sign}' seq {sequences_collected+1}/20"): 
                return False
                
            sequence = []
            for frame_num in range(30): # 30 frames per sequence
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                data = []
                for i in range(2):
                    if results.multi_hand_landmarks and i < len(results.multi_hand_landmarks):
                        for lm in results.multi_hand_landmarks[i].landmark:
                            data.extend([lm.x, lm.y, lm.z])
                    else:
                        data.extend([0] * 63)
                sequence.append(data[:126]) # Ensures exactly 126
                
                if results.multi_hand_landmarks:
                    for hl in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                        
                cv2.rectangle(frame, (0, 0), (640, 60), (0,0,0), -1)
                cv2.putText(frame, f"Recording seq {sequences_collected+1}: Frame {frame_num+1}/30", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("SignBridge Setup", frame)
                if cv2.waitKey(20) & 0xFF == 27: return False
                
            np.save(os.path.join(sign_path, f"{sequences_collected}.npy"), np.array(sequence))
            sequences_collected += 1
    return True

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("Welcome to SignBridge Initial Setup!")
    if not wait_for_key(cap, "Press SPACE to begin Data Collection (ESC to quit)"):
        sys.exit()
        
    print("Starting Static Sign Collection (A-Z)...")
    if not collect_static(cap): sys.exit()
    
    print("Starting Dynamic Sign Collection (Hello, Thank You)...")
    if not collect_dynamic(cap): sys.exit()
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n--- Data Collection Complete! ---")
    print("Training Static Model...")
    subprocess.run([sys.executable, "train_model.py"])
    
    print("Training Dynamic Model...")
    subprocess.run([sys.executable, "train_dynamic_model.py"])
    subprocess.run([sys.executable, "convert_to_onnx.py"])
    
    print("\n✅ Setup Complete! You can now run the app.")
