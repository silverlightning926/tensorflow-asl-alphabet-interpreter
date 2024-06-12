# ASL Alphabet Recognition With Tensorflow
*<div style="color:gray;margin-top:-10px;">By Siddharth Rao</div>*

---

![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=flat&logo=Keras&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white) ![MNIST](https://img.shields.io/badge/Dataset-MNIST_Handwritten_Digits-blue)

---

![Python Lint - autopep8 Workflow](https://github.com/silverlightning926/tensorflow-asl-alphabet-interpreter/actions/workflows/python-lint.yaml/badge.svg)

---

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [ASL Alphabet Recognition With Tensorflow](#asl-alphabet-recognition-with-tensorflow)
  - [Summary](#summary)
  - [Current Status](#current-status)
  - [Getting Started](#getting-started)
  - [Requirments](#requirments)
  - [Linting with Autopep8](#linting-with-autopep8)
  - [Technologies](#technologies)
  - [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
  - [Results](#results)
  - [License](#license)

<!-- /code_chunk_output -->

---

## Summary
This repository contains my code for training and runnning a machine learning model, for classifying images of the American Sign Language (ASL) alphabet. The model was architected and trained using a Google's TensorFlow library.

## Current Status
This project is on hiatus. There may be updates and improvements at a later date, but for now, I have moved on to other projects.

To find the current version of the trained version of the model in the [release selection](https://github.com/silverlightning926/tensorflow-asl-alphabet-interpreter/releases/) or this repository, or use [the version included in this repo](./model.keras).

## Getting Started

To get started with this machine learning model for recognizing handwritten numbers, follow these steps:

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/your-username/tensorflow-mnist.git
    ```

2. Install the required dependencies by running the following command:
    ```bash
    pip install -r requirements.txt
    ```

3. Run `main.py` to download the dataset (if it is not already downloaded) and train and save the model with the data.
    ```bash
    python ./src/main.py
    ```

4. To run the interactive model test after training the model with the steps above, run the following command:
    ```bash
    python ./src/run_model.py
    ```

Feel free to explore the code and make any modifications as needed.

## Requirments
kaggle = 1.6.14
keras = 3.3.3
opencv_python = 4.10.0.82
tensorflow = 2.16.1
These requirements can be found in and downloaded by using [requirements.txt](./requirements.txt)

## Linting with Autopep8
To ensure consistent code formatting, you can use Autopep8, a Python library that automatically formats your code according to the PEP 8 style guide. To install Autopep8, run the following command:
```bash
pip install autopep8
```

Once installed, you can use Autopep8 to automatically format your code by running the following command:
```bash
autopep8 --in-place --recursive ./src
```

This will recursively format all Python files in the current directory and its subdirectories.

Remember to run Autopep8 regularly to maintain a clean and consistent codebase. This repo contains the [Python Lint GitHub Workflow](./.github/workflows/python-lint.yaml) to ensure the repository stays linted.

If you are using VSCode, you can download and the [Autopep8 VSCode Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.autopep8) and add these lines to your `settings.json` to format with Autopep8 automatically as you type and when you save.
```json
"[python]": {
        "editor.formatOnType": true,
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "ms-python.autopep8"
    }
```

## Technologies
The technologies used in this project include but are not limited to:
- Python: The main programming language used for developing the machine learning model and associated scripts.
- TensorFlow: A popular open-source machine learning framework developed by Google, for nueral network and machine learning development.
- Keras: A high-level neural networks API written in Python, used as a user-friendly interface to TensorFlow for building and configuring the model architecture.
- Git: A distributed version control system used for tracking changes and collaborating on the codebase.
- GitHub Actions: A CI/CD platform provided by GitHub, used for automating the linting workflow and displaying the linting badge in the README.
- Autopep8: A Python library used for automatically formatting the code according to the PEP 8 style guide.

## Dataset
The model is currently trained using the [Synthetic ASL Alphabet Dataset](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet) hosted on [Kaggle](https://www.kaggle.com/). This model contains multiple synthetically created images of each ASL alphabet. This model included pictures with multiple backgrounds and lighting conditions. The model contains many pictures for each letter, giving a plentiful amount of data to train on.

Unforturnately, this dataset did come with some downfalls. Due to the images being generated synthetically, the images did not include any camera, compression, or other artifacts that usually appear in real life. This might have contributed to model overfitting. To conteract this, data augementation needed to be added manually. The image was agumented afterwards with noise, a hue delta, and a brightness delta.

The dataset also only incuded pictures of the letter appearing up from the center bottom of the frame, without showing a body or face. This most likely also contributed to overfitting, especially when running the model on other data that doesn't look like the dataset. The fix would be to include other datasets, and this might potentially be done in the future.

## Model Architecture
The model architecture used for the ASL alphabet recognition project is a convolutional neural network (CNN). CNNs are well-suited for image classification tasks due to their ability to automatically learn hierarchical features from the input data.

The architecture consists of multiple layers, including convolutional layers, pooling layers, dropout layers, and fully connected layers. The convolutional layers apply filters to the input image, extracting local features and preserving spatial relationships. The pooling layers downsample the feature maps, reducing the spatial dimensions and extracting the most important features. The fully connected layers connect all the neurons from the previous layer to the next layer, enabling the model to learn complex patterns and make predictions.

The specific architecture used in this project includes multiple convolutional layers with different filter sizes and depths, followed by max pooling layers to reduce the spatial dimensions. The output of the convolutional layers is flattened and passed through fully connected layers with dropout regularization to prevent overfitting. The final layer uses softmax activation to produce the probability distribution over the ASL alphabet classes.

The model architecture is designed to balance complexity and performance, providing a good trade-off between accuracy and computational efficiency. It has been trained on the Synthetic ASL Alphabet Dataset, which contains multiple synthetically created images of each ASL alphabet. However, to further improve the model's performance, additional datasets and data augmentation techniques can be explored in the future.

## Results
After training the model on the datasets, I was able to get the model accuracy around 98% accuracy on images from the dataset.

Unfortunately, running the model on other data such as from a webcam, the accuracy drops quite a bit. It is able to stabilize on some letters at time, but it fluctuated a lot. This is probably due to the model overfitting due to the reasons outlined in the [dataset section](#dataset).

## License
This repository is governed under the MIT license. The repository's license can be found here: [LICENSE](./LICENSE).