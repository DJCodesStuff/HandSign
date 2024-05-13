from hands_package.Build_Model import BuildModel

from flask import Flask, request, jsonify
import json
import cv2
import numpy as np

app = Flask(__name__)
model_builder = BuildModel()  # Instantiate your BuildModel class

# @app.route('/collect_data', methods=['POST'])
# def collect_data():
#     model_builder.collecting_data()
#     return jsonify({'message': 'Data collection completed'})

@app.route('/create_dataset', methods=['POST'])
def create_dataset():
    model_builder.dataset_creation()
    return jsonify({'message': 'Dataset creation completed'}), 200

@app.route('/train_model', methods=['POST'])
def train_model():
    model_builder.training_model()
    return jsonify({'message': 'Model training completed'}), 200

@app.route('/process_frame', methods=['POST'])
def process_frame():
    labels_dict = request.json.get('labels_dict')
    frame = np.array(request.json.get('frame'), dtype=np.uint8)
    sentence = request.json.get('sentence')
    prev_prediction = request.json.get('prev_prediction')
    
    sentence, prev_prediction = model_builder.process_frame(labels_dict, frame, sentence, prev_prediction)
    
    
    return jsonify({'sentence': sentence, 'prev_prediction': prev_prediction}), 200

if __name__ == '__main__':
    app.run(debug=True)

