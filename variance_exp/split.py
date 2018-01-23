from variance_exp import preprocess_clueweb as p
import itertools
import params
import numpy as np


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
        train_queries = set(queries[train])
        subsets_train_queries = []
        for combination in itertools.combinations(train_queries, 150):
            if len(subsets_train_queries) > 30:
                break
            subsets_train_queries.append(combination)
        new_train = []
        for subset in subsets_train_queries:
            for query in subset:
                new_train.extend(np.where(queries == query))
            new_train = np.array(new_train)
            train_file = preprocess.create_train_file(X[new_train], y[new_train], queries[new_train], fold_number)
        test_file = preprocess.create_train_file(X[test], y[test], queries[test], fold_number, True)
        fold_number += 1
