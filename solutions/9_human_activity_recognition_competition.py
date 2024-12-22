#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Parse command-line arguments
parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="human_activity_recognition.model", type=str, help="Model path")


# Dataset class for loading and processing data
class Dataset:
    CLASSES = ["sitting", "sittingdown", "standing", "standingup", "walking"]

    def __init__(self, name="human_activity_recognition.train.csv.xz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print(f"Downloading dataset {name}...", file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=f"{name}.tmp")
            os.rename(f"{name}.tmp", name)

        # Load the dataset and if it contains column "class", split it to `targets`.
        self.data = pd.read_csv(name)
        if "class" in self.data:
            self.target = np.array([Dataset.CLASSES.index(target) for target in self.data["class"]], np.int32)
            self.data = self.data.drop("class", axis=1)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()

        # Split the data into training and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(
            train.data, train.target, test_size=0.2, random_state=args.seed
        )

        # Train a GradientBoostingClassifier with optimized parameters
        model = GradientBoostingClassifier(
            n_estimators=150,  
            max_depth=7,       
            learning_rate=0.1,  
            random_state=args.seed
        )
        model.fit(X_train, y_train)

        # Evaluate the model on the validation set
        y_valid_pred = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, y_valid_pred)
        print(f"Validation accuracy: {accuracy * 100:.2f}%")

        # Serialize and save the model with maximum compression
        with lzma.open(args.model_path, "wb", preset=9) as model_file:
            pickle.dump(model, model_file)

        # Check model size 
        #model_size = os.path.getsize(args.model_path) / 1024 / 1024 
        #print(f"Model size: {model_size:.2f} MiB")

    else:
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Generate predictions for the test set
        predictions = model.predict(test.data)
        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    predictions = main(main_args)

    if predictions is not None:
        for prediction in predictions:
            print(Dataset.CLASSES[prediction])
