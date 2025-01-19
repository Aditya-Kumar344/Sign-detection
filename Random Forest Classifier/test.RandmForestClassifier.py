import cv2
import pickle
import mediapipe as mp
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_data = pickle.load(open('./model.p', 'rb'))
model = model_data['model2']

capture = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

labels_dict = {i: chr(65 + i) for i in range(26)}

expected_feature_size = 84

while True:
  data_aux = []
  x_ = []
  y_ = []

  ret, frame = capture.read()
  if not ret:
    break

  H, W, _ = frame.shape
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  results = hands.process(frame_rgb)
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        z = hand_landmarks.landmark[i].z
        x_.append(x)
        y_.append(y)
        data_aux.append(x - min(x_))
        data_aux.append(y - min(y_))
        data_aux.append(z)

      mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), circle_radius=4),
        mp_drawing.DrawingSpec(color=(0, 0, 255), circle_radius=2)
      )

    data_aux = pad_sequences([data_aux], maxlen=expected_feature_size, dtype='float32')[0]

    if len(data_aux) == expected_feature_size:
      prediction = model.predict([np.asarray(data_aux)])
      prediction_char = labels_dict[int(prediction[0])]

      x1 = int(min(x_) * W) - 10
      y1 = int(min(y_) * H) - 10
      x2 = int(max(x_) * W) + 10
      y2 = int(max(y_) * H) + 10
      cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
      cv2.putText(frame, prediction_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

  cv2.imshow("Sign Detection", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

capture.release()
cv2.destroyAllWindows()