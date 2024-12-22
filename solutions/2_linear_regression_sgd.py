#!/usr/bin/env python3
import argparse

import math
import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization strength")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=92, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[list[float], float, float]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset.
    data, target = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=args.seed)
   

    # DONE: Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    bias = np.ones((100, 1))
    data = np.hstack((data,bias))

    # DONE: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial linear regression weights.
    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)
    number_train_examples = len(train_data)

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        shuffled_train_data = train_data[permutation] # shuffle the input training data
        shuffled_train_target = train_target[permutation] # shuffle the target training data

        # DONE: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `len(train_data)`.

        total_squared_error = 0
        for i in range(0, number_train_examples, args.batch_size):
            batch_data = shuffled_train_data[i:i + args.batch_size]
            batch_target = shuffled_train_target[i:i + args.batch_size]

            predictions = np.dot(batch_data, weights) # dot product of input train data and weights to predict target train data
            errors = predictions - batch_target # difference between the predicted training target data and actual training target data
            total_squared_error += np.sum(np.square(errors)) # sum the squared errors for the entire mini-batch

            # Compute the gradient and update weights
            # The gradient for the input example $(x_i, t_i)$ is
            # - $(x_i^T weights - t_i) x_i$ for the unregularized loss (1/2 MSE loss),
            # - $args.l2 * weights_with_bias_set_to_zero$ for the L2 regularization loss,
            #   where we set the bias to zero because the bias should not be regularized,
            # and the SGD update is
            #   weights = weights - args.learning_rate * gradient
            loss_gradient = np.dot(errors, batch_data) / args.batch_size # compute gradient for unregularised loss
            regularization_gradient = args.l2 * weights # compute gradient for regularisation loss
            regularization_gradient[-1] = 0  # Set bias to zero for regularisation loss
            weights -= args.learning_rate * (loss_gradient + regularization_gradient) # update the weights

        mean_squared_error = total_squared_error / number_train_examples # compute MSE on the entire training data
        rmse_loss_per_epoch = math.sqrt(mean_squared_error) # compute RMSE on the entire training data

        # Compute RMSE for test data
        test_predictions = np.dot(test_data, weights)  # make predictions on the test data
        test_errors = test_predictions - test_target   # difference between the predicted test target data and actual test target data
        test_mse_loss = np.mean(np.square(test_errors))  # compute MSE on test data
        test_rmse_loss = math.sqrt(test_mse_loss)  # compute RMSE on test data

        # DONE: Append current RMSE on train/test to `train_rmses`/`test_rmses`.
        train_rmses.append(rmse_loss_per_epoch)
        test_rmses.append(test_rmse_loss)

    # DONE: Compute into `explicit_rmse` test data RMSE when fitting
    # `sklearn.linear_model.LinearRegression` on `train_data` (ignoring `args.l2`).
    linear_regression_model = sklearn.linear_model.LinearRegression()
    linear_regression_model.fit(train_data, train_target)  # Fit the model on train data

    explicit_test_predictions = linear_regression_model.predict(test_data)  # make predictions on the test data
    explicit_mse_loss = np.mean(np.square(explicit_test_predictions - test_target))  # compute MSE for predicted values and actual test target data
    explicit_rmse = math.sqrt(explicit_mse_loss)  # compute RMSE

    # if args.plot:
    #     import matplotlib.pyplot as plt
    #     plt.plot(train_rmses, label="Train")
    #     plt.plot(test_rmses, label="Test")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("RMSE")
    #     plt.legend()
    #     plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, test_rmses[-1], explicit_rmse

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, sgd_rmse, explicit_rmse = main(main_args)
    print("Test RMSE: SGD {:.3f}, explicit {:.1f}".format(sgd_rmse, explicit_rmse))
    print("Learned weights:", *("{:.3f}".format(weight) for weight in weights[:12]), "...")