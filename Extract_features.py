import cv2
import mediapipe as mp
import numpy as np
import os
from utils import extract_keypoints 

mp_holistic = mp.solutions.holistic

def uniform_sample_or_pad(sequence_data, target_frames=30):
    num_frames = len(sequence_data)
    if num_frames == 0:
        return None  # Safely flag empty videos instead of hardcoding 222 dimensions
    if num_frames >= target_frames:
        indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)
        return sequence_data[indices]
    else:
        padding = target_frames - num_frames
        last_frame = np.repeat(sequence_data[-1:], padding, axis=0)
        return np.concatenate([sequence_data, last_frame], axis=0)

def process_dataset(input_dir, output_dir, target_frames=30):
    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')]
    
    print(f"Starting extraction for {len(video_files)} videos...")
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for video_name in video_files:
            video_path = os.path.join(input_dir, video_name)
            cap = cv2.VideoCapture(video_path)
            sequence_data = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False                  
                results = holistic.process(image)
                keypoints = extract_keypoints(results)
                sequence_data.append(keypoints)
                
            cap.release()
            sequence_data = np.array(sequence_data)
            sequence_data = uniform_sample_or_pad(sequence_data, target_frames=target_frames)
            
            if sequence_data is None:
                print(f"WARNING: Skipping {video_name} - No frames extracted.")
                continue
                
            save_name = os.path.splitext(video_name)[0] + '.npy'
            save_path = os.path.join(output_dir, save_name)
            np.save(save_path, sequence_data)
            print(f"Saved tensor for: {video_name}")

if __name__ == "__main__":
    INPUT_FOLDER = r"C:\Users\PC\Desktop\FinSign's Dataset\Augmented Dataset"
    OUTPUT_FOLDER = r"C:\Users\PC\Desktop\FinSign's Dataset\Extracted Tensors"
    process_dataset(INPUT_FOLDER, OUTPUT_FOLDER, target_frames=30)