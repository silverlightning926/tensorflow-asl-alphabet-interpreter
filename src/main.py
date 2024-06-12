from load_data import getData
from build_model import build_model

from build_model import MODEL_PATH


def main():
    train_data, test_data, _ = getData()

    model = build_model()

    model.fit(train_data, epochs=10, batch_size=32)

    model.evaluate(test_data)

    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()
