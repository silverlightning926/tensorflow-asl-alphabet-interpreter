import tensorflow as tf
from keras.api.models import Sequential


def fit_model(model, train_data: tf.data.Dataset, epochs=10, batch_size=32, verbose=1):
    model.fit(train_data, epochs=epochs,
              batch_size=batch_size, verbose=verbose)

    return model


def evaluate_model(model, test_data: tf.data.Dataset):
    return model.evaluate(test_data)


def save_model(model: Sequential, path: str):
    model.save(path)
