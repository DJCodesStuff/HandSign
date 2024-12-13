# HandSign - Hand Sign Language Recognition API

This project is an API designed to receive frame data (such as hand images) and return the corresponding hand sign language using a machine learning model. The API leverages a pre-built model to interpret hand signs and can be integrated into various applications such as sign language translators or educational tools. The project is still under development, and further improvements are being made.

## Features

- **Hand Sign Language Detection**: The API processes input frames (images) to recognize and return corresponding hand signs.
- **Model Integration**: The API uses a trained model for recognizing hand gestures.
- **Extensible**: The project is designed to be expanded with more features in the future, including improving model accuracy and performance.

## Project Structure

- **`API_Hands.py`**: Contains the main API functionality to receive frame data and return hand sign predictions.
- **`API_Hands_req.py`**: Handles API requests and manages the required inputs and outputs for the hand sign recognition.
- **`buildapimodel.py`**: Used to build and configure the machine learning model for hand sign recognition.
- **`hands_package/`**: This package contains the model building logic (`Build_Model_nn.py`) and initialization scripts.
- **`working/`**: Stores the serialized data and model files, including `data.pickle` and `model.h5`.

## Hand Gesture Recognition: 
This project includes a Python implementation of a hand gesture recognition system built with TensorFlow, MediaPipe, and OpenCV. Here's a breakdown of the functionality:

**Data Collection**: Captures hand gesture images via webcam and organizes them into directories, each representing a gesture class.
**Landmark Extraction**: Uses MediaPipe to detect 2D hand landmarks (21 points per hand) for each image and normalizes the data for consistency.
**Neural Network Model**: Implements a feed-forward neural network with:

Three hidden layers:

**Batch normalization for stability**

**Dropout for regularization**

**Softmax output for classification**

**Training**: Trains the model on the processed dataset and evaluates using accuracy, precision, recall, and AUC metrics.
Real-time Gesture Prediction: Processes video frames to predict gestures on-the-fly, updating predictions dynamically based on detected changes.
Key Features:

**Dynamic Dataset** Handling: The system adapts to the number of gesture classes based on dataset structure.
Real-time Processing: Leverages MediaPipe and TensorFlow to process and classify hand gestures in live video streams.
Customizability: Adjustable parameters for gesture class size, dataset size, and model architecture.
You can use this implementation for gesture-based control systems, sign language translation, or interactive applications!

## Installation

To set up this project locally, follow the steps below:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/HandSign.git
    ```

2. Navigate to the project directory:

    ```bash
    cd HandSign
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Ensure that you have the necessary environment set up for running the API (Python version >= 3.8 is recommended).

## Usage

You can start the API and begin sending requests to recognize hand sign language from images. Here’s how you can run the API locally:

1. Create your own dataset using the script buildapimodel.py. When you start the script, a webcam will window will open up with a propmt to press Q wheneer you are ready, put up the hand sign that you want to be as Label 0 and wait for the script to take rames of the Hand sign. After it takes the frames, it will prompt you again for the next label. This will go on for the number of times mentioned in the script Build_Model_nn.py line 17 variable: "number_of_classes".

```bash
python buildapimodel.py
```

2. After this script finishes, you model is ready with your dataset. Now you can start the API_Hands.py script to deploy the flask api.

```bash
python API_Hands.py
```

3. I have prepared a test script that runs locally where you test the same signs that you used to create your datasets.

```bash
python API_Hands_req.py
```


