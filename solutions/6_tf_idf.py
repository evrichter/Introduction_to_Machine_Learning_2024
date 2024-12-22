#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import sys
import urllib.request
import re
from collections import Counter

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--idf", default=True, action="store_true", help="Use IDF weights")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=79, type=int, help="Random seed")
parser.add_argument("--tf", default=True, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")

# Dataset loader class
class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names


# Main function
def main(args: argparse.Namespace) -> float:
    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    # Create train-test split.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)

    # Tokenization and term extraction
    token_pattern = re.compile(r"\w+")
    term_counts = Counter()

    for document in train_data:
        terms = token_pattern.findall(document)
        term_counts.update(terms)

    # Filter terms occurring at least twice
    terms = {term for term, count in term_counts.items() if count >= 2}
    term_to_index = {term: i for i, term in enumerate(sorted(terms))}
    print("Number of unique terms with at least two occurrences:", len(terms))

    # IDF computation
    idf_weights = None
    if args.idf:
        document_frequencies = np.zeros(len(terms), dtype=np.float32)
        for document in train_data:
            present_terms = set(token_pattern.findall(document))
            for term in present_terms:
                if term in term_to_index:
                    document_frequencies[term_to_index[term]] += 1
        idf_weights = np.log(len(train_data) / (document_frequencies + 1))

    # Feature extraction function
    def extract_features(documents, use_tf, idf_weights=None):
        features = np.zeros((len(documents), len(terms)), dtype=np.float32)
        for i, document in enumerate(documents):
            term_freq = Counter(token_pattern.findall(document))
            for term in term_freq:
                if term in term_to_index:
                    if use_tf:
                        features[i, term_to_index[term]] = term_freq[term]
                    else:
                        features[i, term_to_index[term]] = 1  # Binary indicator
            if use_tf:
                # Normalize term frequencies to sum to 1
                total_count = features[i].sum()
                if total_count > 0:
                    features[i] /= total_count
        if idf_weights is not None:
            features *= idf_weights  # Apply IDF weights
        return features

    # Extract features for train and test sets
    train_features = extract_features(train_data, args.tf, idf_weights)
    test_features = extract_features(test_data, args.tf, idf_weights)

    # Train logistic regression
    model = sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000)
    model.fit(train_features, train_target)

    # Predict on test set
    test_predictions = model.predict(test_features)

    # Evaluate using macro-averaged F1 score
    f1_score = sklearn.metrics.f1_score(test_target, test_predictions, average="macro")
    return 100 * f1_score


# Entry point
if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(main_args)
    print("F-1 score for TF={}, IDF={}: {:.1f}%".format(main_args.tf, main_args.idf, f1_score))
