import cv2
import torch
import threading
import numpy as np
import mediapipe as mp
from utils import extract_keypoints
from Train_model import FinSignTransformer

def format_natural_sentence(buffer):
    if not buffer:
        return ""
        
    raw_sequence = " + ".join(buffer)
    
    # Comprehensive grammar rules for FinSign's banking vocabulary
    grammar_rules = {
        # Teller to Client
        "How can I help you": "How can I help you today?",
        "How can I help you + Purpose of the loan": "How can I help you? What is the purpose of your loan?",
        "Loan + Purpose of the loan": "Could you clarify the purpose of the loan?",
        "Updating Data + ID Number": "We need your National ID number to update your data.",
        "Signature + Cheque": "Please sign the cheque here.",
        "ID Number + Signature": "Please provide your ID number and signature.",
        "How Much + Money": "How much money would you like to process?",
        
        # Client to Teller
        "Withdraw + 20000 L.E": "I want to withdraw 20,000 L.E.",
        "Withdraw + 100K L.E": "I want to withdraw 100,000 L.E.",
        "Deposit + Cheque": "I want to deposit a cheque.",
        "Deposit + Money": "I want to make a cash deposit.",
        "Card + Visa Renewal": "I need to renew my Visa card.",
        "Bank Transfer + 20000 L.E": "I want to transfer 20,000 L.E.",
        "Bank Transfer + 100K L.E": "I want to transfer 100,000 L.E.",
        "Bank Transfer + Money": "I need to make a bank transfer."
    }
    
    # If the exact sequence isn't in the dictionary, gracefully fallback to space-separated words
    return grammar_rules.get(raw_sequence, " ".join(buffer))

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

class CamoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
                    self.new_frame_event.set()

    def read_new(self):
        self.new_frame_event.wait(timeout=0.1)
        with self.lock:
            self.new_frame_event.clear()
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

classes = np.load('classes.npy', allow_pickle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FinSignTransformer(input_size=222, num_classes=len(classes)).to(device)
model.load_state_dict(torch.load('finsign_transformer_model.pth', map_location=device))
model.eval()

sequence = []
ema_probs = None
EMA_ALPHA = 0.70 
current_prediction = "IDLE"
missing_hands_grace = 0
frame_counter = 0

# Sentence Tracking Variables
sentence_buffer = []
last_stable_word = ""
cooldown_counter = 0
candidate_word = ""
candidate_count = 0

print("Starting synchronized camera stream...")
stream = CamoStream(src=0).start()
print("Starting AI Translation...")

with mp_holistic.Holistic(min_detection_confidence=0.35, min_tracking_confidence=0.35) as holistic:
    while True:
        ret, frame = stream.read_new()
        if not ret or frame is None:
            continue
            
        frame_counter += 1
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
        keypoints = extract_keypoints(results)
        hands_active = (results.left_hand_landmarks is not None) or (results.right_hand_landmarks is not None)

        if frame_counter % 2 == 0:
            if hands_active:
                missing_hands_grace = 0
                sequence.append(keypoints)
                sequence = sequence[-30:]
            else:
                missing_hands_grace += 1
                if missing_hands_grace > 5:
                    sequence.clear()
                    ema_probs = None
                    current_prediction = "IDLE (No Hands)"
                    cooldown_counter = 20  
                elif len(sequence) > 0:
                    sequence.append(sequence[-1])
                    sequence = sequence[-30:]

        if len(sequence) == 30 and missing_hands_grace <= 5:
            seq_array = np.array(sequence, dtype=np.float32)
            input_tensor = torch.from_numpy(seq_array).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
                
                if ema_probs is None:
                    ema_probs = probabilities
                else:
                    ema_probs = EMA_ALPHA * probabilities + (1 - EMA_ALPHA) * ema_probs
                
                predicted_idx = int(np.argmax(ema_probs))
                confidence = float(ema_probs[predicted_idx])
                predicted_label = classes[predicted_idx]
                
                prob_str = " | ".join([f"{classes[i]}: {ema_probs[i]*100:.0f}%" for i in range(len(classes))])
                print(f"\r[{prob_str}] -> Pred: {predicted_label}   ", end="")
                
                if confidence > 0.85:  
                    current_prediction = f"{predicted_label} ({confidence*100:.1f}%)"
                    
                    if predicted_label.upper() != "IDLE":
                        if missing_hands_grace == 0:
                            if predicted_label == candidate_word:
                                candidate_count += 1
                            else:
                                candidate_word = predicted_label
                                candidate_count = 1
                            
                            if candidate_count >= 5:
                                if predicted_label != last_stable_word or cooldown_counter > 15:
                                    sentence_buffer.append(predicted_label)
                                    last_stable_word = predicted_label
                                    cooldown_counter = 0
                                    candidate_count = 0  
                    else:
                        cooldown_counter += 1
                        candidate_count = 0  
                else:
                    current_prediction = "IDLE"
                    cooldown_counter += 1
                    candidate_count = 0
                            
        cv2.rectangle(frame, (0,0), (1280, 80), (24, 24, 24), -1)
        cv2.putText(frame, f"Prediction: {current_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        display_text = format_natural_sentence(sentence_buffer[-6:]) 
        cv2.putText(frame, f"Sentence: {display_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
        cv2.imshow('FinSign Pipeline A: AI Translation', frame)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence_buffer.clear()
            last_stable_word = ""

stream.stop()
cv2.destroyAllWindows()