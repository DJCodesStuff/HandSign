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
- **`hands_package/`**: This package contains the model building logic (`Build_Model.py`) and initialization scripts.
- **`working/`**: Stores the serialized data and model files, including `data.pickle` and `model.p`.

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

1. Run the API script:

    ```bash
    python API_Hands.py
    ```

2. Send image frame data (e.g., hand images) to the API for recognition. The API will return the corresponding hand sign based on the model’s prediction.

### Example Request

Here’s an example of how to send a request to the API using Python:

```python
import requests

# URL of the API endpoint
url = 'http://localhost:5000/predict'

# Image data to send (replace with actual image data)
image_data = {'frame': open('hand_image.jpg', 'rb')}

# Send the request to the API
response = requests.post(url, files=image_data)

# Print the API response (hand sign prediction)
print(response.json())
