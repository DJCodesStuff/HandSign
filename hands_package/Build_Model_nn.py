import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall, AUC
DATA_DIR_ = 'working/data'

class BuildModel:
    DATA_DIR = DATA_DIR_
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if len([d for d in os.listdir(DATA_DIR_) if os.path.isdir(os.path.join(DATA_DIR_, d))]) == 26:
        # Dynamically set number_of_classes based on directories in DATA_DIR
        number_of_classes = len([d for d in os.listdir(DATA_DIR_) if os.path.isdir(os.path.join(DATA_DIR_, d))])
    else:
        number_of_classes = 3 # for testing purposes, change here to make larger dataset
    

    dataset_size = 100

    static_image_mode = True
    min_detection_confidence = 0.8
    max_num_hands = 2  # Can handle up to 2 hands now

    data = []
    labels = []
    capture = 0

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=static_image_mode, 
                           min_detection_confidence=min_detection_confidence, 
                           max_num_hands=max_num_hands)
    
    @staticmethod
    def pad_sequences(data, max_len):
        """
        Pads each sequence in the data to the max_len with zeros.
        :param data: List of sequences (arrays) with varying lengths
        :param max_len: Maximum length to pad the sequences to
        :return: Numpy array of padded sequences
        """
        padded_data = []
        for seq in data:
            padded_seq = list(seq)  # Ensure the sequence is a list
            # Pad with zeros if the sequence is shorter than max_len
            if len(padded_seq) < max_len:
                padded_seq.extend([0] * (max_len - len(padded_seq)))
            padded_data.append(padded_seq)
        return np.array(padded_data)

    @classmethod
    def collecting_data(cls):
        cap = cv2.VideoCapture(cls.capture)

        for j in range(cls.number_of_classes):
            if not os.path.exists(os.path.join(cls.DATA_DIR, str(j))):
                os.makedirs(os.path.join(cls.DATA_DIR, str(j)))

            print('Collecting data for class {}'.format(j))

            done = False
            while True:
                ret, frame = cap.read()
                cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                            cv2.LINE_AA)
                cv2.imshow('frame', frame)
                if cv2.waitKey(25) == ord('q'):
                    break

            counter = 0
            while counter < cls.dataset_size:
                ret, frame = cap.read()
                cv2.imshow('frame', frame)
                cv2.waitKey(25)
                cv2.imwrite(os.path.join(cls.DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

                counter += 1

        cap.release()
        cv2.destroyAllWindows()

    @classmethod
    def dataset_creation(cls):
        for dir_ in os.listdir(cls.DATA_DIR):
            if dir_ == ".DS_Store":
                continue

            for img_path in os.listdir(os.path.join(cls.DATA_DIR, dir_)):
                data_aux = []
                x_ = []
                y_ = []

                img = cv2.imread(os.path.join(cls.DATA_DIR, dir_, img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = cls.hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        hand_data = []
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y

                            x_.append(x)
                            y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            hand_data.append(x - min(x_))
                            hand_data.append(y - min(y_))

                        data_aux.extend(hand_data)  # Combine data from multiple hands

                    cls.data.append(data_aux)
                    cls.labels.append(dir_)

        with open('working/data.pickle', 'wb') as f:
            pickle.dump({'data': cls.data, 'labels': cls.labels}, f)

        print("Directories:")
        for dir_ in os.listdir(cls.DATA_DIR):
            print(dir_)

    @classmethod
    def training_model(cls):
        # Load data and labels from pickle
        data_dict = pickle.load(open('working/data.pickle', 'rb'))
        raw_data = data_dict['data']
        labels = np.asarray(data_dict['labels'])

        # Determine the maximum length of any sequence in raw_data
        max_length = max(len(seq) for seq in raw_data)

        # Pad sequences to have consistent length
        data = cls.pad_sequences(raw_data, max_length)

        # Convert labels to numerical values if needed
        unique_labels = np.unique(labels)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        labels = np.array([label_map[label] for label in labels])
        
        # One-hot encode labels if using categorical crossentropy
        labels = to_categorical(labels)

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

        # Define the neural network model
        model = Sequential([
            # First layer
            Dense(256, input_shape=(x_train.shape[1],), activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second layer
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Third layer
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Output layer
            Dense(len(unique_labels), activation='softmax')
        ])
        

        # Compile the model with a suitable optimizer, loss function, and metrics
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',        # Standard accuracy
                Precision(name='precision'),  # Precision metric
                Recall(name='recall'),        # Recall metric
                AUC(name='auc')              # Area Under the ROC Curve
                ]
        )

        # Train the model
        model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

        
        # Evaluate the model
        evaluation_results = model.evaluate(x_test, y_test)
        loss = evaluation_results[0]  # First value is the loss
        accuracy = evaluation_results[1]  # Second value is accuracy
        precision = evaluation_results[2]  # Third value is precision
        recall = evaluation_results[3]  # Fourth value is recall
        auc = evaluation_results[4]  # Fifth value is AUC

        # Print results
        print(f'Loss: {loss:.4f}')
        print(f'Accuracy: {accuracy * 100:.2f}%')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'AUC: {auc:.4f}')

        # Save the model
        model.save('working/model.h5')

    @classmethod
    def process_frame(cls, labels_dict, frame, sentence, prev_prediction, w, h):
        """
        Processes a single video frame to predict gestures from one or two hands.
        """
        data_aux = []
        x_ = []
        y_ = []

        # Load the trained model
        model = load_model('working/model.h5')

        # Convert the frame to the correct format
        frame_rgb = np.array(frame, dtype="uint8").reshape((w, h, 3))
        results = cls.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    hand_data.append(x - min(x_))
                    hand_data.append(y - min(y_))

                data_aux.extend(hand_data)  # Concatenate data from multiple hands

            if data_aux:
                # Pad the input to ensure consistent length if only one hand is detected
                max_features = 42 * 2  # Assuming 21 landmarks per hand and x/y coordinates
                if len(data_aux) < max_features:
                    data_aux.extend([0] * (max_features - len(data_aux)))

                # Make prediction
                prediction = model.predict(np.array([data_aux]))
                predicted_character = labels_dict[str(np.argmax(prediction))]

                if prev_prediction != predicted_character:
                    sentence += " " + predicted_character
                    prev_prediction = predicted_character

        return sentence, prev_prediction




