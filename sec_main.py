import cv2 as cv
import numpy as np
import time

img = cv.imread('test_pics/selfie640.png')
k = 100

model = YuNet(modelPath='face_detection_yunet_2022mar.onnx',
              inputSize=[320, 320],
              confThreshold=0.9,
              nmsThreshold=0.3,
              topK=5000,
              backendId=3,
              targetId=0)
h, w, _ = img.shape
# Inference
model.setInputSize([w, h])
tic = time.perf_counter()
for i in range(1, k):
    results = model.infer(img)
toc = time.perf_counter()