from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from pygame import mixer
from imutils.video import VideoStream
import tensorflow as tf
import numpy as np
import argparse
import imutils
import time
import cv2
import os

mixer.init()
sound1 = mixer.Sound('output.wav')
sound2 = mixer.Sound('sound.wav')

def mask_detection_prediction(frame, faceNet, maskNet):

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")

        maskNet.set_tensor(maskNet.get_input_details()[0]['index'], faces)
        maskNet.invoke()

        preds = maskNet.get_tensor(maskNet.get_output_details()[0]['index'])

    return (locs, preds)


prototxtPath = os.path.join("face_detector", "deploy.prototxt")
weightsPath = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = tf.lite.Interpreter(model_path="fmd_model.tflite")
maskNet.allocate_tensors()

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

i = 0

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    (locs, preds) = mask_detection_prediction(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        mask = pred[0]
        withoutMask = pred[1]

        if mask > withoutMask:
            if i % 80 == 0:
                sound1.play()
            label = "Mask"
            color = (0, 255, 0)
            print("Normal")
        else:
            if i % 80 == 0:
                sound2.play()
            label = "No Mask"
            color = (0, 0, 255)
            print("Alert!!!")
        i+=1

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
