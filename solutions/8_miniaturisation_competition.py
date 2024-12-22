import argparse
import lzma
import os
import pickle
import sys
import numpy as np
import sklearn.neural_network
import urllib.request
import numpy.typing as npt
from typing import Optional
from scipy.ndimage import rotate, shift

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--model_path", default="miniaturization.model", type=str, help="Model path")


class MLPFullDistributionClassifier(sklearn.neural_network.MLPClassifier):
    class FullDistributionLabels:
        y_type_ = "multiclass"

        def fit(self, y):
            return self

        def transform(self, y):
            return y

        def inverse_transform(self, y):
            return np.argmax(y, axis=-1)

    def _validate_input(self, X, y, incremental, reset):
        X, y = self._validate_data(X, y, multi_output=True, dtype=(np.float64, np.float32), reset=reset)
        if (not hasattr(self, "classes_")) or (not self.warm_start and not incremental):
            self._label_binarizer = self.FullDistributionLabels()
            self.classes_ = y.shape[1]
        return X, y


class Dataset:
    def __init__(self, name="mnist.train.npz", data_size=None, url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print(f"Downloading dataset {name}...", file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=f"{name}.tmp")
            os.rename(f"{name}.tmp", name)

        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28 * 28]).astype(float)


def augment_data(data, labels):
    augmented_data, augmented_labels = [], []
    for image, label in zip(data, labels):
        image_reshaped = image.reshape(28, 28)
        augmented_data.append(image)
        augmented_labels.append(label)

        # Rotate image
        rotated = rotate(image_reshaped, angle=15, reshape=False).flatten()
        augmented_data.append(rotated)
        augmented_labels.append(label)

        # Shift image
        shifted = shift(image_reshaped, shift=(2, 2)).flatten()
        augmented_data.append(shifted)
        augmented_labels.append(label)

    return np.array(augmented_data), np.array(augmented_labels)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    np.random.seed(args.seed)

    if args.predict is None:
        # Training phase
        train = Dataset()
        
        # Augment the training data
        train.data, train.target = augment_data(train.data, train.target)

        alpha = 0.8  # Balance soft and hard targets
        temperature = 10  # Smoother teacher predictions

        # Teacher model
        teacher = MLPFullDistributionClassifier(
            hidden_layer_sizes=(2048, 1024, 512),
            activation="relu",
            max_iter=150,  # Increase iterations
            random_state=args.seed,
            early_stopping=True,
            validation_fraction=0.1,
            alpha=1e-4,  # L2 regularization
        )

        # Train teacher
        teacher.fit(train.data, np.eye(10)[train.target])

        # Distillation process
        soft_targets = teacher.predict_proba(train.data)
        soft_targets_temp = np.exp(np.log(soft_targets) / temperature)
        soft_targets_temp /= np.sum(soft_targets_temp, axis=1, keepdims=True)

        student_targets = (
            alpha * np.eye(10)[train.target] + (1 - alpha) * soft_targets_temp
        )

        # Student model
        student = MLPFullDistributionClassifier(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            max_iter=200,
            random_state=args.seed,
            early_stopping=True,
            validation_fraction=0.1,
            solver="adam",  # Use Adam optimizer
            alpha=1e-4,  # L2 regularization
        )

        # Train student with distillation
        student.fit(train.data, student_targets)

        # Apply compression to student model
        student._optimizer = None
        for i in range(len(student.coefs_)):
            student.coefs_[i] = student.coefs_[i].astype(np.float16)
        for i in range(len(student.intercepts_)):
            student.intercepts_[i] = student.intercepts_[i].astype(np.float16)

        # Save the model
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(student, model_file)

    else:
        # Prediction phase
        test = Dataset(args.predict)

        # Load the model
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Make predictions
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
