from hands_package.Build_Model import BuildModel
import cv2

labels_dict = {0: 'A', 1: 'B', 2: 'C'} # import from new file

# Example usage:
sentence = ""
prev_prediction = ""
cap = cv2.VideoCapture(0)
counter = 0
old_sentence = ""

while True:
    ret, frame = cap.read()
    sentence, prev_prediction = BuildModel.process_frame(frame = frame, labels_dict=labels_dict, sentence=sentence, prev_prediction=prev_prediction)
    if old_sentence != sentence:
        print(sentence)
        counter += 1
    
    if counter > 100:
        counter = 0
        sentence = " "

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()