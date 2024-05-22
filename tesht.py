from hands_package.Build_Model import BuildModel
import numpy as np
import cv2

labels_dict = {0: 'A', 1: 'B', 2: 'C'}  # Import from your new file
sentence = ""
prev_prediction = ""
counter = 0
old_sentence = ""


cap = cv2.VideoCapture(0)
c = 0 

while True:
    ret, frame = cap.read()
    c += 1
    sentence, prev_prediction = BuildModel.process_frame(frame.tolist(), labels_dict, sentence, prev_prediction)
    # if old_sentence != sentence:
    print(sentence, prev_prediction, counter)
    counter += 1

    
    if counter > 30:
        counter = 0
        sentence = " "
        break

    if cv2.waitKey(1) & 0xFF == ord('q') or c == 3:
        break

cap.release()
cv2.destroyAllWindows()

print(sentence, prev_prediction)