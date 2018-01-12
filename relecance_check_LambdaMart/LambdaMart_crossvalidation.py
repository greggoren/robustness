import preprocess_clueweb as p
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
    number_of_queries = len(set(queries))
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict()
    evaluator.remove_score_file_from_last_run()
    folds = preprocess.create_folds(X, y, queries, 5)
    fold_number = 1
    trees = int(sys.argv[1])
    leaves = int(sys.argv[2])
    model_handler = mh.model_handler_LambdaMart(trees, leaves)
    for train, test in folds:
        model_handler.set_queries_to_folds(queries, test, fold_number)
        train_file = preprocess.create_train_file(X[train], y[train], queries[train])
        test_file = preprocess.create_train_file_cv(X[test], y[test], queries[test], fold_number, True)
        trec = model_handler.fit_model_on_train_set_and_run(train_file, test_file, test, queries, evaluator)
        fold_number += 1
    final = evaluator.order_trec_file(trec)
    evaluator.run_trec_eval_on_test_correlation(final, trees, leaves)
