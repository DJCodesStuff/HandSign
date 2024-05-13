import cv2
import numpy as np
import requests

labels_dict = {0: 'A', 1: 'B', 2: 'C'}  # Import from your new file
sentence = ""
prev_prediction = ""
counter = 0
old_sentence = ""


def train_model():
    url = 'http://127.0.0.1:5000/train_model'  # Update with your API URL

    response = requests.post(url)

    if response.status_code == 200:
        data = response.json()
        message = data.get('message')
        print(f'Message: {message}')
    else:
        print(f'Error: {response.status_code} - {response.text}')
        return None, None


def process_frame(frame, labels_dict, sentence, prev_prediction):
    url = 'http://127.0.0.1:5000/process_frame'  # Update with your API URL
    payload = {
        'labels_dict': labels_dict,
        'frame': frame.tolist(),  # Convert NumPy array to list
        'sentence': sentence,
        'prev_prediction': prev_prediction
    }
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()
        updated_sentence = data.get('sentence')
        updated_prev_prediction = data.get('prev_prediction')
        return updated_sentence, updated_prev_prediction
    else:
        print(f'Error: {response.status_code} - {response.text}')
        return None, None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    sentence, prev_prediction = process_frame(frame, labels_dict, sentence, prev_prediction)
    if old_sentence != sentence:
        print(sentence)
        counter += 1
    
    if counter > 100:
        counter = 0
        sentence = " "

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

