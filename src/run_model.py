import tensorflow as tf

import cv2

from keras.api.models import load_model, Sequential

from load_data import DOWNSCALED_IMAGE_SIZE
from build_model import MODEL_PATH


model: Sequential = load_model(MODEL_PATH)


def preprocessImage(image):
    image = cv2.resize(image, DOWNSCALED_IMAGE_SIZE)
    image = image / 255.0
    image = image.reshape(1, *DOWNSCALED_IMAGE_SIZE, 3)
    return image


def predict(image):
    return model.predict(image)


# TODO: Get Actual Class Labels
def getLabel(prediction):
    return chr(ord('A') + tf.argmax(prediction[0]).numpy())


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    image = preprocessImage(frame)
    prediction = predict(image)
    label = getLabel(prediction)

    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('ASL Alphabet Recognition', image)

    if (cv2.waitKey(1) & 0xFF == ord('q')) or cv2.getWindowProperty('ASL Alphabet Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
