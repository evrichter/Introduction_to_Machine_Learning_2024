#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import numpy.typing as npt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")


class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in the range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # scale pixel values 
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train.data / 255.0)

        # define MLP model with hyperparameters
        model = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),  
            activation="relu",                  
            solver="adam",                      
            alpha=0.001,                         
            batch_size=128,                     
            learning_rate_init=0.001,           
            max_iter=10,                         
            early_stopping=True,                 
            validation_fraction=0.1,            
            random_state=args.seed
        )

        # Train the model
        model.fit(X_train, train.target)

        # Calculate and print training accuracy
        train_predictions = model.predict(X_train)
        train_accuracy = accuracy_score(train.target, train_predictions)
        print(f"Training accuracy: {train_accuracy * 100:.2f}%")

        # If you trained one or more models, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained `MLPClassifier` is in the `model` variable.
        model._optimizer = None
        for i in range(len(model.coefs_)):
            model.coefs_[i] = model.coefs_[i].astype(np.float16)
        for i in range(len(model.intercepts_)): 
            model.intercepts_[i] = model.intercepts_[i].astype(np.float16)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump({'model': model, 'scaler': scaler}, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        # Load the model and scaler
        with lzma.open(args.model_path, "rb") as model_file:
            model_data = pickle.load(model_file)
            model = model_data['model']   # This accesses the model correctly
            scaler = model_data['scaler'] # This accesses the scaler

        # Scale test data and make predictions
        X_test_scaled = scaler.transform(test.data / 255.0)
        predictions = model.predict(X_test_scaled)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)