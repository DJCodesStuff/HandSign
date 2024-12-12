from hands_package.Build_Model_nn import BuildModel
import cv2

model = BuildModel()

# model.collecting_data() # uncomment this line to create new dataset
model.dataset_creation()
model.training_model()

