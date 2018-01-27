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
    number_of_queries = len(set(queries))
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict()
    evaluator.remove_score_file_from_last_run()

    # trees = int(sys.argv[1])
    # leaves = int(sys.argv[2])

    trees = [(i + 1) * 10 for i in range(39, 45)]
    leaves = [(1 + i) * 5 for i in range(25, 30)]
    models = zip(leaves, trees)
    for leaf, tree in models:
        fold_number = 1
        model_handler = mh.model_handler_LambdaMart(leaf, tree)
        folds = preprocess.create_folds(X, y, queries, 5)
        for train, test in folds:
            model_handler.set_queries_to_folds(queries, test, fold_number)
            train_file = "features" + str(fold_number)
            test_file = "features_test" + str(fold_number)
            trec = model_handler.fit_model_on_train_set_and_run(train_file, test_file, test, queries, evaluator,
                                                                fold_number)
            fold_number += 1
        final = evaluator.order_trec_file(trec)
        evaluator.run_trec_eval_on_test_correlation(final, tree, leaf)
