from variance_exp_lb import preprocess_clueweb as p
from variance_exp_lb import LambdaMart_models_handler as mh
from variance_exp_lb import evaluator as e
from variance_exp_lb import params
import sys
from multiprocessing import Pool
import pickle
import functools
import time
def get_results(score_file, test_indices):
    results = {}
    with open(score_file) as scores:
        for index, score in enumerate(scores):
            results[test_indices[index]] = score.split()[-1].rstrip()
    return results


def update_scores(results, scores, subset, tree, leaves):
    for index in results:
        scores[(tree, leaves)][index].append((subset, results[index]))
    return scores
if __name__ == "__main__":
    start = time.time()
    preprocess = p.preprocess()
    X, y, queries = preprocess.retrieve_data_from_file(params.data_set_file, params.normalized)
    number_of_queries = len(set(queries))
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict()
    evaluator.remove_score_file_from_last_run()
    folds = preprocess.create_folds(X, y, queries, 5)
    fold_number = 1
    trees = 250
    leaves = [(i + 1) * 10 for i in range(17, 26)]
    scores = {(trees, leaf): {i: [] for i in range(len(queries))} for leaf in leaves}

    for leaf in leaves:
        for train, test in folds:
            model_handler = mh.model_handler_LambdaMart(trees, leaf)
            with Pool(processes=5) as pool:
                model_handler.set_queries_to_folds(queries, test, fold_number)
                f = functools.partial(model_handler.fit_model_on_train_set_for_variance, params.qrels, fold_number)
                score_files = pool.map(f, range(1, 31))
                for score in score_files:
                    subset = int(score.split("#")[1])
                    results = get_results(score, test)
                    scores = update_scores(results, scores, subset, trees, leaf)
            fold_number += 1
    print("it took:", time.time() - start)
    with open("variance_data1", 'wb') as data:
        pickle.dump(scores, data)
