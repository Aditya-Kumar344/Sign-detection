import os
import cv2
import pickle
import mediapipe as mp
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

Data = './data'
Labelled = './labelled_data'

os.makedirs(Labelled, exist_ok = True)

data = []
labels = []

for directory in os.listdir(Data):
  dir_path = os.path.join(Data, directory)
  if not os.path.isdir(dir_path):
    continue
      
  labelled_dir = os.path.join(Labelled, directory)
  os.makedirs(labelled_dir, exist_ok=True)
  
  for img_name in os.listdir(dir_path):
    data_aux = []
    x_ = []
    y_ = []
    
    img_path = os.path.join(dir_path, img_name)
    img = cv2.imread(img_path)
    if img is None:
      print(f"Failed to load image: {img_path}")
      continue
        
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_img)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        for i in range(len(hand_landmarks.landmark)):
          x = hand_landmarks.landmark[i].x
          y = hand_landmarks.landmark[i].y
          x_.append(x)
          y_.append(y)
        
        for i in range(len(hand_landmarks.landmark)):
          x = hand_landmarks.landmark[i].x
          y = hand_landmarks.landmark[i].y
          data_aux.append(x - min(x_))
          data_aux.append(y - min(y_))
        
        mp_drawing.draw_landmarks(
          img,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style()
        )
      
      data.append(data_aux)
      labels.append(directory)
      
      output_path = os.path.join(labelled_dir, img_name)
      cv2.imwrite(output_path, img)
            
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Processed {len(data)} images")
print(f"Found {len(set(labels))} unique gestures")