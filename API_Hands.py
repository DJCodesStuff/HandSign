from hands_package.Build_Model_nn import BuildModel
from flask import Flask, request, jsonify
import numpy as np
import base64

app = Flask(__name__)
model_builder = BuildModel()  # Instantiate your BuildModel class

@app.route('/test_connection', methods=['POST'])
def test_connection():
    return jsonify({'message': 'Connection successful'}), 200

@app.route('/create_dataset', methods=['POST'])
def create_dataset():
    model_builder.dataset_creation()
    return jsonify({'message': 'Dataset creation completed'}), 200

@app.route('/train_model', methods=['POST'])
def train_model():
    model_builder.training_model()
    return jsonify({'message': 'Model training completed'}), 200

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    try:
        labels_dict = {'0': 'A', '1': 'B', '2': 'C'}  # Adjust based on your dataset
        frame = request.json.get('frame')  # Frame sent as a list or bytes (ensure correct type)
        sentence = request.json.get('sentence', "")
        prev_prediction = request.json.get('prev_prediction', "")
        width = request.json.get('width')
        height = request.json.get('height')

        sentence, prev_prediction = BuildModel.process_frame(
            labels_dict=labels_dict,
            frame=frame,
            sentence=sentence,
            prev_prediction=prev_prediction,
            w=width,
            h=height
        )

        return jsonify({'sentence': sentence, 'prev_prediction': prev_prediction}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

        
    

if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=True)
    app.run(host='0.0.0.0',port = 6969, debug=True) # local testing





