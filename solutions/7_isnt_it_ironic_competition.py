#!/usr/bin/env python3

import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV



parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")


class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            for line in dataset_file:
                label, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.target.append(int(label))
        self.target = np.array(self.target, np.int32)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:

    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()

        param_grid = {
            'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],  #unigrams and bigrams
            'tfidfvectorizer__max_df': [0.85, 0.90], 
            'tfidfvectorizer__min_df': [2, 3],  # remove rare terms
            'logisticregression__C': [0.1, 1, 5],  # regularise
            'logisticregression__penalty': ['l2'],
            'logisticregression__solver': ['liblinear'],
        }

        model = Pipeline([
            ('tfidfvectorizer', TfidfVectorizer(stop_words='english', sublinear_tf=True)),
            ('logisticregression', LogisticRegression(random_state=args.seed))
        ])

        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(train.data, train.target)
        print("Best Parameters:", grid_search.best_params_)

        best_model = grid_search.best_estimator_
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(best_model, model_file)

    else:
        test = Dataset(args.predict)
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)


        # Predictions on test data
        predictions = model.predict(test.data)
        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)