import cv2
import numpy as np
import requests

labels_dict = {'0': 'A', '1': 'B', '2': 'C'}  # Import from your new file
sentence = " "
prev_prediction = " "
counter = 0



def train_model():
    # url = 'http://127.0.0.1:5000/train_model'  # Update with your API URL
    url = "https://handsign-02e8.onrender.com/train_model"


    response = requests.post(url)

    if response.status_code == 200:
        data = response.json()
        message = data.get('message')
        print(f'Message: {message}')
    else:
        print(f'Error: {response.status_code} - {response.text}')
        return None, None
    
def test_connection():
    # url = 'http://127.0.0.1:6969/test_connection'  # Update with your API URL
    url = "https://handsign-02e8.onrender.com/test_connection"


    response = requests.post(url)

    if response.status_code == 200:
        data = response.json()
        message = data.get('message')
        print(f'Message: {message}')
    else:
        print(f'Error: {response.status_code} - {response.text}')
        return None, None


def process_frame(frame, sentence, prev_prediction, w, h):
    url = 'http://127.0.0.1:6969/process_frame'  # Update with your API URL

    payload = {
        'frame': frame.tolist(),  # Convert NumPy array to list
        'sentence': sentence,
        'prev_prediction': prev_prediction,
        'width': w,
        'height': h
    }
    headers = {'Content-Type': 'application/json'}
    # print(payload)
    

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
    # print(frame.shape)

    width = 720  # Replace with actual image width
    height = 1280 # Replace with actual image height

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = frame_rgb.flatten()
    sentence, prev_prediction = process_frame(frame = frame_rgb, sentence = sentence, prev_prediction = prev_prediction, w = width, h = height)
    print(sentence, prev_prediction, counter)
    counter += 1

    
    if counter > 100:
        counter = 0
        sentence = " "
        break


cap.release()
cv2.destroyAllWindows()

# print(sentence, prev_prediction)

# train_model()
# predict_shit()
# test_connection()