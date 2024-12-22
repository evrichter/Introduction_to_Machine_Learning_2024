#!/usr/bin/env python3
import argparse
import warnings

import numpy as np

import sklearn.datasets
import sklearn.exceptions
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bootstrap_samples", default=1000, type=int, help="Bootstrap resamplings")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=47, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[list[tuple[float, float]], float]:
    # Suppress warnings about the solver not converging
    warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Load the digits dataset
    data = sklearn.datasets.load_digits(n_class=args.classes)

    # Split the data into train and test sets
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data.data, data.target, test_size=args.test_size, random_state=args.seed)

    # Train two logistic regression models with polynomial features of degree 1 and 2
    models = []
    for d in [1, 2]:
        model = sklearn.pipeline.Pipeline([
            ("features", sklearn.preprocessing.PolynomialFeatures(degree=d)),
            ("estimator", sklearn.linear_model.LogisticRegression(solver="saga", random_state=args.seed, max_iter=100))
        ])
        model.fit(train_data, train_target)
        models.append(model)

    # Precompute test set predictions to save time
    test_predictions = [model.predict(test_data) for model in models]

    # Compute bootstrap resampling for accuracies
    scores = np.zeros((2, args.bootstrap_samples))
    for i in range(args.bootstrap_samples):
        indices = generator.choice(len(test_data), size=len(test_data), replace=True)
        for j in range(2):
            scores[j, i] = np.mean(test_predictions[j][indices] == test_target[indices]) * 100

    # Compute 95% confidence intervals
    confidence_intervals = [
        (np.percentile(scores[0], 2.5), np.percentile(scores[0], 97.5)),
        (np.percentile(scores[1], 2.5), np.percentile(scores[1], 97.5))
    ]

    # Compute paired bootstrap differences and the probability of the null hypothesis
    score_differences = scores[1] - scores[0]
    result = np.mean(score_differences <= 0) * 100

    # Optional: Plot histograms and results if requested
    if args.plot:
        import matplotlib.pyplot as plt
        def histogram(ax, data, color=None):
            ax.hist(data, bins=int(round((np.max(data) - np.min(data)) * len(test_data) / 100)) + 1,
                    weights=100 * np.ones_like(data) / len(data), color=color)

        plt.figure(figsize=(12, 5))
        ax = plt.subplot(121)
        for score, ci, color in zip(scores, confidence_intervals, ["#d00", "#0d0"]):
            histogram(ax, score, color + "8")
            ax.axvline(np.mean(score), ls="-", color=color, label=f"mean: {np.mean(score):.1f}%")
            ax.axvline(ci[0], ls="--", color=color, label="95% CI")
            ax.axvline(ci[1], ls="--", color=color)
        ax.set_xlabel("Model accuracy")
        ax.set_ylabel("Frequency [%]")
        ax.legend()

        ax = plt.subplot(122)
        histogram(ax, score_differences)
        for percentile in [1, 2.5, 5, 25, 50, 75, 95, 97.5, 99]:
            value = np.percentile(score_differences, percentile)
            color = {1: "#f00", 2.5: "#d60", 5: "#dd0", 25: "#0f0", 50: "#000"}[min(percentile, 100 - percentile)]
            ax.axvline(value, ls="--", color=color, label=f"{percentile:04.1f}%: {value:.1f}")
        ax.axvline(0, ls="--", color="#f0f", label=f"{result:.2f}% null hypothesis")
        ax.set_xlabel("Model accuracy difference")
        ax.set_ylabel("Frequency [%]")
        ax.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return confidence_intervals, result


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    confidence_intervals, result = main(main_args)
    print("Confidence intervals of the two models:")
    for confidence_interval in confidence_intervals:
        print(f"- [{confidence_interval[0]:.2f}% .. {confidence_interval[1]:.2f}%]")
    print(f"The estimated probability that the null hypothesis holds: {result:.2f}%")
