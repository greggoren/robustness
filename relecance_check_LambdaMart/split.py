from relecance_check_LambdaMart import preprocess_clueweb as p
from relecance_check_LambdaMart import LambdaMart_models_handler as mh
from relecance_check_LambdaMart import evaluator as e
import params
import sys


def get_results(score_file, test_indices):
    results = {}
    with open(score_file) as scores:
        for index, score in enumerate(scores):
            results[test_indices[index]] = score
    return results


if __name__ == "__main__":
    preprocess = p.preprocess()
    X, y, queries = preprocess.retrieve_data_from_file(params.data_set_file, params.normalized)
    folds = preprocess.create_folds(X, y, queries, 5)
    fold_number = 1
    for train, test in folds:
        train_file = preprocess.create_train_file(X[train], y[train], queries[train], fold_number)
        test_file = preprocess.create_train_file(X[test], y[test], queries[test], fold_number, True)
        fold_number += 1
