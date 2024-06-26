import tensorflow as tf

import cv2

from keras.api.models import load_model, Sequential

from load_data import DOWNSCALED_IMAGE_SIZE, getData
from build_model import MODEL_PATH


model: Sequential = load_model(MODEL_PATH)


def getClassLabels():
    return getData()[2]


classLabels = getClassLabels()


def preprocessImage(image):
    image = cv2.resize(image, DOWNSCALED_IMAGE_SIZE)
    image = image / 255.0
    image = image.reshape(1, *DOWNSCALED_IMAGE_SIZE, 3)
    return image


def predict(image):
    return model.predict(image)


def getLabel(prediction):
    return classLabels[prediction.argmax()]


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    image = cv2.flip(frame, 1)

    image = preprocessImage(frame)
    prediction = predict(image)
    label = getLabel(prediction)

    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('ASL Alphabet Recognition', frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')) or cv2.getWindowProperty('ASL Alphabet Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
