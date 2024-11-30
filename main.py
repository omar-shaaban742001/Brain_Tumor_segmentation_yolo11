from ultralytics import YOLO
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

model_path = "F:/programming/computer vision nanodegree/projects/brain_tumor/Brain_Tumor_segmentaion_yolo11/model/segmentaion_yolo11.pt"
model = YOLO(model_path)


image_path = 'F:/programming/computer vision nanodegree/projects/brain_tumor/Brain_Tumor_segmentaion_yolo11/images/1.jpg'
img = cv.imread(image_path)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


result = model.predict(image_path)
result[0].show()

plt.imshow(img)
plt.show()