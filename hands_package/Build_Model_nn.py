import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

class BuildModel:
    DATA_DIR = 'working/data'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    number_of_classes = 3 # change number here to add more labels for creating dataset.
    dataset_size = 100

    static_image_mode = True
    min_detection_confidence = 0.8
    max_num_hands = 1

    data = []
    labels = []
    # capture = cv2.VideoCapture(0)
    capture = 0

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8, max_num_hands=1)
    hands = mp_hands.Hands(static_image_mode=static_image_mode, 
                                min_detection_confidence=min_detection_confidence, 
                                max_num_hands=max_num_hands)

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
        # mp_hands = mp.solutions.hands
        # mp_drawing = mp.solutions.drawing_utils
        # mp_drawing_styles = mp.solutions.drawing_styles

        # hands = mp_hands.Hands(static_image_mode=cls.static_image_mode, 
        #                         min_detection_confidence=cls.min_detection_confidence, 
        #                         max_num_hands=cls.max_num_hands)

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
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y

                            x_.append(x)
                            y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    cls.data.append(data_aux)
                    cls.labels.append(dir_)

        f = open('working/data.pickle', 'wb')
        pickle.dump({'data': cls.data, 'labels': cls.labels}, f)
        f.close()

        print("Directories:")
        for dir_ in os.listdir(cls.DATA_DIR):
            print(dir_)

    @classmethod
    def training_model(cls):
        data_dict = pickle.load(open('working/data.pickle', 'rb'))
        data = np.asarray(data_dict['data'])
        labels = np.asarray(data_dict['labels'])

        # Convert labels to numerical values if needed
        unique_labels = np.unique(labels)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        labels = np.array([label_map[label] for label in labels])
        
        # One-hot encode labels if using categorical crossentropy
        labels = to_categorical(labels)

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
        print((x_train.shape[1],))

        # Define the neural network model
        model = Sequential([
            Dense(128, input_shape=(x_train.shape[1],), activation='relu'),
            Dense(64, activation='relu'),
            Dense(len(unique_labels), activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

        # Evaluate the model
        loss, accuracy = model.evaluate(x_test, y_test)
        print(f'{accuracy * 100:.2f}% of samples were classified correctly!')

        # Save the model using Keras
        model.save('working/model.h5')


    @classmethod
    def process_frame(cls, labels_dict, frame, sentence, prev_prediction, w, h):
        data_aux = []
        x_ = []
        y_ = []

        # Load the trained model
        model = load_model('working/model.h5')

        frame_rgb = np.array(frame, dtype="uint8").reshape((w, h, 3))
        results = cls.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            if data_aux:
                prediction = model.predict(np.array([data_aux]))
                predicted_character = labels_dict[str(np.argmax(prediction))]

                if prev_prediction != predicted_character:
                    sentence += " " + predicted_character
                    prev_prediction = predicted_character

        return sentence, prev_prediction