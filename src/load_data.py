import tensorflow as tf

from kaggle.api.kaggle_api_extended import KaggleApi
import os

from keras.api.utils import image_dataset_from_directory

TRAIN_DATA_PATH = './data/Train_Alphabet/'
TEST_DATA_PATH = './data/Test_Alphabet/'

RAW_IMAGE_SIZE = (512, 512)
DOWNSCALED_IMAGE_SIZE = (126, 126)
BATCH_SIZE = 32

api = KaggleApi()


def _authenticate():
    api.authenticate()


def _download_data(dataset: str, path: str = 'data'):
    api.dataset_download_files(dataset, path=path, unzip=True)


def _fetchData():
    if os.path.exists('data'):
        print('Data already downloaded. Skipping download.')
        return

    _authenticate()
    _download_data('lexset/synthetic-asl-alphabet')
    print('Data downloaded successfully.')


def _loadData():
    train_data = image_dataset_from_directory(
        TRAIN_DATA_PATH,
        image_size=RAW_IMAGE_SIZE,
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        class_names=['A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                     'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                     'U', 'V', 'W', 'X', 'Y', 'Z']
    )

    test_data = image_dataset_from_directory(
        TEST_DATA_PATH,
        image_size=RAW_IMAGE_SIZE,
        label_mode='categorical',
        batch_size=BATCH_SIZE
    )

    return train_data, test_data, train_data.class_names


def _preprocessImage(image_data, label_data):
    image_data = tf.image.resize(image_data, DOWNSCALED_IMAGE_SIZE)
    image_data = tf.cast(image_data, tf.float32) / 255.0
    return image_data, label_data


def _preprocessDataset(train_data: tf.data.Dataset, test_data: tf.data.Dataset):
    train_data = train_data.map(
        _preprocessImage,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    test_data = test_data.map(
        _preprocessImage,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    return train_data, test_data


def _cacheData(train_data: tf.data.Dataset, test_data: tf.data.Dataset):
    train_data = train_data.cache().prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    test_data = test_data.cache().prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    return train_data, test_data


def getData():
    _fetchData()

    train_data, test_data, class_names = _loadData()
    train_data, test_data = _preprocessDataset(train_data, test_data)
    train_data, test_data = _cacheData(train_data, test_data)

    return train_data, test_data, class_names
