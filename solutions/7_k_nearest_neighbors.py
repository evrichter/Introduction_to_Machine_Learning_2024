#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request

import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--k", default=1, type=int, help="K nearest neighbors to consider")
parser.add_argument("--p", default=2, type=int, help="Use L_p as distance metric")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
parser.add_argument("--weights", default="uniform", choices=["uniform", "inverse", "softmax"], help="Weighting to use")
# If you add more arguments, ReCodEx will keep them with your default values.


class MNIST:
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


def main(args: argparse.Namespace) -> float:
    mnist = MNIST(data_size=args.train_size + args.test_size)
    mnist.data = sklearn.preprocessing.MinMaxScaler().fit_transform(mnist.data)
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        mnist.data, mnist.target, test_size=args.test_size, random_state=args.seed)

    def compute_distance(x1, x2, p):
        return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

    def predict(test_sample):
        distances = []
        for train_sample in train_data:
            distance = compute_distance(test_sample, train_sample, args.p)
            distances.append(distance)

        distances = np.array(distances)

        nearest_indices = np.argsort(distances)[:args.k]

        nearest_classes = train_target[nearest_indices]
        nearest_distances = distances[nearest_indices]

        if args.weights == "uniform":
            weights = np.ones_like(nearest_distances)
        elif args.weights == "inverse":
            weights = 1 / (nearest_distances + 1e-8)
        elif args.weights == "softmax":
            weights = np.exp(-nearest_distances)
            weights /= np.sum(weights)

        votes = {}
        for cls, weight in zip(nearest_classes, weights):
            if cls not in votes:
                votes[cls] = 0
            votes[cls] += weight

        max_weight = -float('inf')
        best_class = None

        for cls, weight in votes.items():
            if weight > max_weight or (weight == max_weight and cls < best_class):
                max_weight = weight
                best_class = cls

        return best_class

    test_predictions = []
    for test_sample in test_data:
        prediction = predict(test_sample)
        test_predictions.append(prediction)

    test_predictions = np.array(test_predictions)

    accuracy = sklearn.metrics.accuracy_score(test_target, test_predictions)

    if args.plot:
        import matplotlib.pyplot as plt
        examples = [[] for _ in range(10)]
        for i in range(len(test_predictions)):
            if test_predictions[i] != test_target[i] and not examples[test_target[i]]:
                nearest_indices = np.argsort([compute_distance(test_data[i], train_sample, args.p) for train_sample in train_data])[:args.k]
                examples[test_target[i]] = [test_data[i], *train_data[nearest_indices]]
        examples = [[img.reshape(28, 28) for img in example] for example in examples if example]
        examples = [[example[0]] + [np.zeros_like(example[0])] + example[1:] for example in examples]
        plt.imshow(np.concatenate([np.concatenate(example, axis=1) for example in examples], axis=0), cmap="gray")
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return 100 * accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(main_args)
    print("K-nn accuracy for {} nearest neighbors, L_{} metric, {} weights: {:.2f}%".format(
        main_args.k, main_args.p, main_args.weights, accuracy))
