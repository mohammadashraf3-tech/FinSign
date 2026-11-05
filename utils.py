import mediapipe as mp
import numpy as np

VITAL_FACE_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    46, 53, 52, 65, 55,
    276, 283, 282, 295, 285
]

def normalize_hand(hand_landmarks, anchor, scale_x, scale_y):
    hand = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark])
    
    # 1. Convert hand coordinates to aspect-corrected space using face scale_x and scale_y
    hand_corr = np.zeros_like(hand)
    hand_corr[:, 0] = (hand[:, 0] - anchor[0]) / scale_x
    hand_corr[:, 1] = (hand[:, 1] - anchor[1]) / scale_y
    hand_corr[:, 2] = hand[:, 2] / ((scale_x + scale_y) / 2.0)

    wrist_corr = hand_corr[0].copy()

    # 2. Internal hand size (distance from wrist 0 to middle finger knuckle 9 in corrected space)
    hand_size = np.linalg.norm(hand_corr[9, :2] - wrist_corr[:2])
    if hand_size < 1e-4:
        hand_size = 1.0

    out_hand = np.zeros_like(hand)
    # Fingers (1..20) strictly relative to wrist, scaled by hand_size
    out_hand[1:] = (hand_corr[1:] - wrist_corr) / hand_size

    # Wrist (0) relative to face anchor, with clamped vertical drop for standing vs sitting
    out_hand[0, 0] = wrist_corr[0]
    out_hand[0, 1] = np.tanh(wrist_corr[1] * 0.35)
    out_hand[0, 2] = wrist_corr[2] * 0.2
    
    return out_hand

def extract_keypoints(results):
    if results.face_landmarks:
        face = np.array([[results.face_landmarks.landmark[idx].x, 
                          results.face_landmarks.landmark[idx].y, 
                          results.face_landmarks.landmark[idx].z] for idx in VITAL_FACE_INDICES])
        anchor = np.mean(face, axis=0)
        
        scale_x = np.abs(face[22, 0] - face[27, 0])
        scale_y = np.abs(face[5, 1] - (face[22, 1] + face[27, 1]) / 2.0)
        if scale_x < 1e-4: scale_x = 0.1
        if scale_y < 1e-4: scale_y = 0.1
    else:
        face = np.zeros((len(VITAL_FACE_INDICES), 3))
        anchor = np.array([0.5, 0.3, 0.0])
        scale_x, scale_y = 0.15, 0.15

    if results.left_hand_landmarks:
        lh = normalize_hand(results.left_hand_landmarks, anchor, scale_x, scale_y)
    else:
        lh = np.zeros((21, 3))

    if results.right_hand_landmarks:
        rh = normalize_hand(results.right_hand_landmarks, anchor, scale_x, scale_y)
    else:
        rh = np.zeros((21, 3))

    if results.face_landmarks:
        face[:, 0] = (face[:, 0] - anchor[0]) / scale_x
        face[:, 1] = (face[:, 1] - anchor[1]) / scale_y
        face[:, 2] = 0.0

    all_points = np.vstack([lh, rh, face])
    return all_points.flatten()