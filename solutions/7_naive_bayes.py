#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter of our NB classifier")
parser.add_argument("--naive_bayes_type", default="gaussian", choices=["gaussian", "multinomial", "bernoulli"])
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=72, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed
    )

    target_counts = np.bincount(train_target)
    total_targets = len(train_target)
    class_prior = target_counts / total_targets

    if args.naive_bayes_type == "gaussian":
        means = []
        variances = []
        for c in range(args.classes):
            class_data = train_data[train_target == c]
            class_mean = class_data.mean(axis=0)
            class_variance = class_data.var(axis=0) + args.alpha
            means.append(class_mean)
            variances.append(class_variance)
        means = np.array(means)
        variances = np.array(variances)

        log_probs = []
        for c in range(args.classes):
            feature_log_prob = scipy.stats.norm.logpdf(test_data, loc=means[c], scale=np.sqrt(variances[c]))
            sum_log_prob = np.sum(feature_log_prob, axis=1)
            class_log_prior = np.log(class_prior[c])
            total_log_prob = sum_log_prob + class_log_prior
            log_probs.append(total_log_prob)
        log_probs = np.array(log_probs).T

    elif args.naive_bayes_type == "multinomial":
        feature_sums = []
        for c in range(args.classes):
            class_data = train_data[train_target == c]
            class_feature_sum = class_data.sum(axis=0)
            feature_sums.append(class_feature_sum)
        feature_sums = np.array(feature_sums)
        
        feature_totals = np.sum(feature_sums, axis=1, keepdims=True)
        feature_probs = (feature_sums + args.alpha) / (feature_totals + args.alpha * train_data.shape[1])

        log_probs = []
        for c in range(args.classes):
            feature_log_prob = test_data * np.log(feature_probs[c])
            sum_log_prob = np.sum(feature_log_prob, axis=1)
            class_log_prior = np.log(class_prior[c])
            total_log_prob = sum_log_prob + class_log_prior
            log_probs.append(total_log_prob)
        log_probs = np.array(log_probs).T

    elif args.naive_bayes_type == "bernoulli":
        binarized_train_data = (train_data >= 8).astype(int)

        feature_probs = []
        for c in range(args.classes):
            class_data = binarized_train_data[train_target == c]
            class_feature_sum = class_data.sum(axis=0)
            class_feature_prob = (class_feature_sum + args.alpha) / (class_data.shape[0] + 2 * args.alpha)
            feature_probs.append(class_feature_prob)
        feature_probs = np.array(feature_probs)

        binarized_test_data = (test_data >= 8).astype(int)
        
        log_probs = []
        for c in range(args.classes):
            positive_log_prob = binarized_test_data * np.log(feature_probs[c])
            negative_log_prob = (1 - binarized_test_data) * np.log(1 - feature_probs[c])
            feature_log_prob = positive_log_prob + negative_log_prob
            sum_log_prob = np.sum(feature_log_prob, axis=1)
            class_log_prior = np.log(class_prior[c])
            total_log_prob = sum_log_prob + class_log_prior
            log_probs.append(total_log_prob)
        log_probs = np.array(log_probs).T

    predictions = np.argmax(log_probs, axis=1)

    correct_predictions = (predictions == test_target)
    test_accuracy = np.mean(correct_predictions)

    test_log_probability = 0
    for i in range(len(test_target)):
        true_class = test_target[i]
        true_log_prob = log_probs[i, true_class]
        test_log_probability += true_log_prob

    return 100 * test_accuracy, test_log_probability

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy, test_log_probability = main(main_args)

    print("Test accuracy {:.2f}%, log probability {:.2f}".format(test_accuracy, test_log_probability))