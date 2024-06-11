from kaggle.api.kaggle_api_extended import KaggleApi
import os

api = KaggleApi()


def _authenticate():
    api.authenticate()


def _download_data(dataset: str, path: str = 'data'):
    api.dataset_download_files(dataset, path=path, unzip=True)


def loadData():
    if os.path.exists('data'):
        print('Data already downloaded. Skipping download.')
        return

    _authenticate()
    _download_data('lexset/synthetic-asl-alphabet')
    print('Data downloaded successfully.')
