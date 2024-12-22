#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=5, type=int, help="Number of classes to use")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.



def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial classification dataset.
    data, target_list = sklearn.datasets.make_multilabel_classification(
        n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
        return_indicator=False, random_state=args.seed)

    # TODO: The `target` is a list of classes for every input example. Convert
    # it to a dense representation (n-hot encoding) -- for each input example,
    # the target should be vector of `args.classes` binary indicators.
    target = np.zeros((args.data_size, args.classes), dtype=int)
    
    for sample, label in enumerate(target_list):
        target[sample, label] = 1  
    
    # Append a constant feature with value 1 to the end of all input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)
    
    number_train_examples = len(train_data)

    # Generate initial model weights.
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        shuffled_train_data = train_data[permutation]
        shuffled_train_target = train_target[permutation]

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        
        for i in range(0, number_train_examples, args.batch_size):
            batch_data = shuffled_train_data[i:i + args.batch_size]
            batch_target = shuffled_train_target[i:i + args.batch_size]

            # Predict using current weights (using sigmoid activation).
            predictions = 1 / (1 + np.exp(-batch_data @ weights))

            # Compute gradient for binary cross-entropy loss
            gradient = batch_data.T @ (predictions - batch_target) / args.batch_size

            # Update weights using the average gradient.
            weights -= args.learning_rate * gradient

        # TODO: After the SGD epoch, compute the micro-averaged and the
        # macro-averaged F1-score for both the train test and the test set.
        # Compute these scores manually, without using `sklearn.metrics`.
        # Calculate training and testing metrics
        
        # Predictions for train and test sets
        train_predictions = 1 / (1 + np.exp(-train_data @ weights))
        test_predictions = 1 / (1 + np.exp(-test_data @ weights))

        train_f1_micro, train_f1_macro = calculate_f1(train_predictions, train_target)
        test_f1_micro, test_f1_macro = calculate_f1(test_predictions, test_target)

        print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
            epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))

    return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]

def calculate_f1(predictions, targets):

    pred_binary = (predictions >= 0.5).astype(int)
    true_positive = np.sum(pred_binary * targets, axis=0)
    false_positive = np.sum(pred_binary * (1 - targets), axis=0)
    false_negative = np.sum((1 - pred_binary) * targets, axis=0)

    f1_scores = (2 * true_positive) / (2  * true_positive + false_positive + false_negative)

    # Macro-averaged F1 
    macro_f1 = np.mean(f1_scores)

    # Micro-averaged F1 
    total_true_positive = np.sum(true_positive)
    total_false_positive = np.sum(false_positive)
    total_false_negative = np.sum(false_negative)
    micro_f1 = (2 * total_true_positive) / (2  * total_true_positive + total_false_positive + total_false_negative)

    return micro_f1, macro_f1


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(main_args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")