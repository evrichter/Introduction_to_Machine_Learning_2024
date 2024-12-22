#!/usr/bin/env python3
import argparse
import warnings

import numpy as np
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--random_samples", default=1000, type=int, help="Number of random samples")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=47, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Suppress warnings about solver convergence
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Load the digits dataset.
    data = load_digits(n_class=args.classes)

    # Split the data randomly into train and test sets
    train_data, test_data, train_target, test_target = train_test_split(
        data.data, data.target, test_size=args.test_size, random_state=args.seed
    )

    # Train linear regression models with polynomial features
    models = []
    for d in [1, 2]:
        pipeline = Pipeline([
            ("features", PolynomialFeatures(degree=d)),
            ("estimator", LogisticRegression(solver="saga", random_state=args.seed, max_iter=100)),
        ])
        pipeline.fit(train_data, train_target)
        models.append(pipeline)

    # Predict the test set for both models
    model0_predictions = models[0].predict(test_data)
    model1_predictions = models[1].predict(test_data)

    # Compute the accuracy of the second model
    second_model_accuracy = np.mean(model1_predictions == test_target) * 100

    # Generate random assignments in smaller chunks to save memory
    scores = []
    chunk_size = 100  # Process 100 samples at a time
    for _ in range(0, args.random_samples, chunk_size):
        chunk_size = min(chunk_size, args.random_samples - len(scores))  # Adjust final chunk size
        assignments = generator.choice(2, size=(chunk_size, len(test_data)))
        combined_predictions = np.where(assignments == 0,
                                         model0_predictions[None, :],
                                         model1_predictions[None, :])
        chunk_scores = np.mean(combined_predictions == test_target, axis=1) * 100
        scores.extend(chunk_scores)

    # Perform one-sided random permutation test and estimate its p-value
    scores = np.array(scores)
    p_value = np.mean(scores >= second_model_accuracy) * 100

    # Plot the results if requested
    if args.plot:
        import matplotlib.pyplot as plt
        bin_size = 100 / len(test_data)
        plt.hist(scores, bins=int((np.max(scores) - np.min(scores)) / bin_size),
                 range=(np.min(scores) - bin_size / 2, np.max(scores) - bin_size / 2),
                 weights=100 * np.ones_like(scores) / len(scores))
        plt.axvline(second_model_accuracy, color="magenta", linestyle="--", label=f"Second model accuracy: {second_model_accuracy:.2f}%")
        plt.xlabel("Permuted model accuracy")
        plt.ylabel("Frequency [%]")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return p_value


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    p_value = main(main_args)
    print(f"The estimated p-value of the random permutation test: {p_value:.2f}%")
