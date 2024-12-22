#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=13, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.80, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Load the diabetes dataset.
    dataset = sklearn.datasets.load_diabetes()
    X = dataset.data
    y = dataset.target

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size=args.test_size, random_state=args.seed)

    rmses = []
    lambdas = np.geomspace(0.01, 10, num=500)

    for l in lambdas:

    # TODO: Using `sklearn.linear_model.Ridge`, fit the train set using
    # L2 regularization, employing the above defined lambdas.
    # For every model, compute the root mean squared error and return the
    # lambda producing lowest RMSE and the corresponding RMSE.
        regularised_model = sklearn.linear_model.Ridge(alpha = l)
        regularised_model.fit(X_train, y_train)
        y_pred = regularised_model.predict(X_test)
        rmse = sklearn.metrics.root_mean_squared_error(y_test, y_pred)

        my_dict = {
            'lambdaKey': l,
            'rmseKey': rmse,
        }

        rmses.append(my_dict)

    # Find the dictionary with the lowest RMSE
    best_rmse_dict = min(rmses, key=lambda x: x['rmseKey'])

    # Extract the best lambda and corresponding RMSE
    best_lambda = best_rmse_dict['lambdaKey']
    best_rmse = best_rmse_dict['rmseKey']

    if args.plot:
        # This block is not required to pass in ReCodEx; however, it is useful
        # to learn to visualize the results. If you collect the respective
        # results for `lambdas` to an array called `rmses`, the following lines
        # will plot the result if you add `--plot` argument.
        import matplotlib.pyplot as plt

        rmsesForPloting = []
        for rmse in rmses:
            temporaryRmse = rmse['rmseKey']
            rmsesForPloting.append(temporaryRmse)

        plt.plot(lambdas, rmsesForPloting)
        plt.xscale("log")
        plt.xlabel("L2 regularization strength $\\lambda$")
        plt.ylabel("RMSE")
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return best_lambda, best_rmse


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    best_lambda, best_rmse = main(main_args)
    print("{:.2f} {:.2f}".format(best_lambda, best_rmse))