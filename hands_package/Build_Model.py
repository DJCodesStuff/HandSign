import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class BuildModel:
    DATA_DIR = 'working/data'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    number_of_classes = 3
    dataset_size = 100

    static_image_mode = True
    min_detection_confidence = 0.8
    max_num_hands = 1

    data = []
    labels = []
    # capture = cv2.VideoCapture(0)
    capture = 0

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
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=cls.static_image_mode, 
                                min_detection_confidence=cls.min_detection_confidence, 
                                max_num_hands=cls.max_num_hands)

        for dir_ in os.listdir(cls.DATA_DIR):
            if dir_ == ".DS_Store":
                continue

            for img_path in os.listdir(os.path.join(cls.DATA_DIR, dir_)):
                data_aux = []
                x_ = []
                y_ = []

                img = cv2.imread(os.path.join(cls.DATA_DIR, dir_, img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = hands.process(img_rgb)
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

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

        model = RandomForestClassifier(max_depth=3, n_jobs=-1)
        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        score = accuracy_score(y_predict, y_test)

        print('{}% of samples were classified correctly !'.format(score * 100))

        f = open('working/model.p', 'wb')
        pickle.dump({'model': model}, f)
        f.close()

    @classmethod
    def process_frame(cls, labels_dict, frame, sentence, prev_prediction):
        data_aux = []
        x_ = []
        y_ = []
        H, W, _ = frame.shape

        model_dict = pickle.load(open('working/model.p', 'rb'))
        model = model_dict['model']

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8, max_num_hands=1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

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

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Perform prediction if there's data
            if data_aux:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[str(prediction[0])]

                # Append predicted character to the sentence
                if prev_prediction != predicted_character:
                    sentence += " "+predicted_character
                    prev_prediction = predicted_character

            # Display sentence in the corner of the frame
            # cv2.putText(frame, sentence, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw rectangle and character on hand region
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            # cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # cv2.imshow('frame', frame)
        
        return sentence, prev_prediction