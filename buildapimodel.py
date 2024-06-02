from hands_package.Build_Model import BuildModel
import cv2

model = BuildModel()

# model.collecting_data()
model.dataset_creation()
model.training_model()

