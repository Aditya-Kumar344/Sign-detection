import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

dataset = pickle.load(open('./data.pickle', 'rb'))

try:
  data = pad_sequences(dataset['data'], padding='post', dtype='float32')
  print(f"Data successfully padded to shape: {data.shape}")
except Exception as e:
  print(f"Error during padding: {e}")
  data = np.array([np.ravel(item) for item in dataset['data']])
  print(f"Data flattened to shape: {data.shape}")

labels = np.asarray(dataset['labels'])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

accuracy = accuracy_score(y_predict, y_test)

with open("Models_Accuracy.txt", 'a') as a:
  a.write(f"\nAccuracy of KNN Model = {(accuracy * 100):.2f}%\n")

with open('model.p', 'wb') as f:
  pickle.dump({'model4': model}, f)