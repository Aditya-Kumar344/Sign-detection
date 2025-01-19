import cv2
import os
import time

data_dir = './data'
if not os.path.exists(data_dir):
  os.makedirs(data_dir)

letters = 26
images = 1000

capture = cv2.VideoCapture(0)

for j in range(letters):
  class_dir = os.path.join(data_dir, str(j))
  if not os.path.exists(class_dir):
    os.makedirs(class_dir)

  print(f"Collecting data for letter {chr(j+65)}")

  image_no = 0
  capturing = False

  while True:
    ret, frame = capture.read()
    if not ret:
      break

    cv2.putText(frame, 'Press Space to capture', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)

    if key == ord(' '): 
      capturing = True
      print("Started capturing images for class {}".format(j))

    if capturing and image_no < images:
      img_path = os.path.join(class_dir, '{}.jpg'.format(image_no))
      cv2.imwrite(img_path, frame)
      print(f"Saved image {img_path}")
      image_no += 1
      time.sleep(0.0125)

    if image_no >= images:
      capturing = False
      print(f"Finished capturing for class {j}")
      break

    if key == ord('q'):
      capture.release()
      cv2.destroyAllWindows()
      exit()

capture.release()
cv2.destroyAllWindows()