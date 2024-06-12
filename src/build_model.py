from keras.api.models import Sequential
from keras.api.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.api.models import load_model

import os

from load_data import DOWNSCALED_IMAGE_SIZE

MODEL_PATH = 'model.keras'


def _build_model():

    if os.path.exists(MODEL_PATH):
        print('Model already exists. Loading model.')
        model = load_model(MODEL_PATH)

    else:

        model = Sequential([
            Input(shape=(*DOWNSCALED_IMAGE_SIZE, 3)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(27, activation='softmax')
        ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_model():
    return _build_model()
