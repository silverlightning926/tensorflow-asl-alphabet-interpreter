from keras.api.models import Sequential
from keras.api.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.api.models import load_model

import os

from main import MODEL_PATH


def _build_model():

    if os.path.exists(MODEL_PATH):
        print('Model already exists. Loading model.')
        model = load_model(MODEL_PATH)

    else:

        model = Sequential([
            Input(shape=(512, 512, 3)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
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
