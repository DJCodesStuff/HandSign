from hands_package.Build_Model import BuildModel
import numpy as np
import cv2



def tesht_simple():

    labels_dict = {'0': 'A', '1': 'B', '2': 'C'}  # Import from your new file
    sentence = ""
    prev_prediction = ""
    counter = 0

    cap = cv2.VideoCapture(0)
    c = 0 

    while True:
        ret, frame = cap.read()
        c += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(frame_rgb.shape)
        frame_rgb = frame_rgb.flatten()
        

        # Assuming frame_rgb_flattened is your flattened list and you know the original width and height
        width = 720  # Replace with actual image width
        height = 1280 # Replace with actual image height

        # # Reshape the flattened list into a NumPy array with the original dimensions (BGR order)
        # frame_rgb = np.array(frame_rgb).reshape((width, height, 3))
        # print(frame_rgb.shape)

        sentence, prev_prediction = BuildModel.process_frame(labels_dict = labels_dict, frame = frame_rgb.tolist(), sentence = sentence, prev_prediction = prev_prediction, w = width, h = height)

        print(sentence, prev_prediction, counter)


        # break

        
        if counter > 300:
            counter = 0
            sentence = " "
            break


    cap.release()
    cv2.destroyAllWindows()

    print(sentence, prev_prediction)

if __name__ == "__main__":
    tesht_simple()