from variance_exp import preprocess_clueweb as p
from variance_exp import LambdaMart_models_handler as mh
from variance_exp import evaluator as e
from variance_exp import params
import sys
from multiprocessing import Pool
import pickle
import functools
import time
def get_results(score_file, test_indices):
    results = {}
    with open(score_file) as scores:
        for index, score in enumerate(scores):
            results[test_indices[index]] = score
    return results


def update_scores(results, scores, subset):
    for index in results:
        scores[subset][index].append(results[index])
    return scores
if __name__ == "__main__":
    start = time.time()
    preprocess = p.preprocess()
    X, y, queries = preprocess.retrieve_data_from_file(params.data_set_file, params.normalized)
    scores = {subset: {i: [] for i in range(len(queries))} for subset in range(31)}
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
        p = Pool(5)
        model_handler.set_queries_to_folds(queries, test, fold_number)
        f = functools.partial(model_handler.fit_model_on_train_set_for_variance, params.qrels, fold_number)
        score_files = p.map(f, range(31))
        for score in score_files:
            subset = int(score.split("#")[1])
            results = get_results(score, test)
            scores = update_scores(results, scores, subset)
        fold_number += 1
    print("it took:", time.time() - start)
    with open("variance_data", 'wb') as data:
        pickle.dump(scores, data)
