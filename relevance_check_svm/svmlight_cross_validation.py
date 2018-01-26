from relevance_check_svm import preprocess_clueweb as p
from relevance_check_svm import evaluator as e
from relevance_check_svm import params
import numpy as np
import os
import subprocess
from multiprocessing import Pool
from functools import partial

def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    return iter(p.stdout.readline, b'')


def learn_svm(C, train_file, fold):
    if not os.path.exists("models/" + str(fold)):
        try:
            os.makedirs("models/" + str(fold))
        except:
            print("weird behaviour")
            print(C, train_file, fold)

    learning_command = "./svm_rank_learn -c " + str(C) + " " + train_file + " " + "models/" + str(
        fold) + "/svm_model" + str(C) + ".txt"
    for output_line in run_command(learning_command):
        print(output_line)
    return "models/" + str(fold) + "/svm_model" + str(C) + ".txt"


def run_svm(C, model_file, test_file, fold):
    score_path = "scores/" + str(fold)
    if not os.path.exists(score_path):
        try:
            os.makedirs(score_path)
        except:
            print("collition")
    rank_command = "./svm_rank_classify " + test_file + " " + model_file + " " + score_path + "/" + str(C)
    for output_line in run_command(rank_command):
        print(output_line)
    return score_path + "/" + str(C)


def retrieve_scores(test_indices, score_file):
    with open(score_file) as scores:
        results = {test_indices[i]: score.rstrip() for i, score in enumerate(scores)}
        return results


def recover_model(model):
    indexes_covered = []
    weights = []
    with open(model) as model_file:
        for line in model_file:
            if line.__contains__(":"):
                wheights = line.split()
                wheights_length = len(wheights)

                for index in range(1, wheights_length - 1):

                    feature_id = int(wheights[index].split(":")[0])
                    if index < feature_id:
                        for repair in range(index, feature_id):
                            if repair in indexes_covered:
                                continue
                            weights.append(0)
                            indexes_covered.append(repair)
                    weights.append(float(wheights[index].split(":")[1]))
                    indexes_covered.append(feature_id)
    return np.array(weights)


def f(train_file, test_file, fold_number, C):
    model_file = learn_svm(C, train_file, fold_number)
    score_file = run_svm(C, model_file, test_file, fold_number)
    return score_file

if __name__ == "__main__":
    preprocess = p.preprocess()
    C_array = [0.1, 1.0, 0.01]
    # C_array = [float(i + 1) / 1000 for i in range(10)]
    # C_array.extend([float(i + 1) / 100 for i in range(10)])
    # C_array.extend([float(i + 1) / 10 for i in range(10)])
    # C_array.extend([float(i + 1) for i in range(10)])
    # C_array.extend([float(i + 1) * 10 for i in range(10)])
    # C_array.extend([float(i + 1) * 100 for i in range(10)])
    X, y, queries = preprocess.retrieve_data_from_file(params.data_set_file, params.normalized)
    number_of_queries = len(set(queries))
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict()
    evaluator.remove_score_file_from_last_run()
    folds = preprocess.create_folds(X, y, queries, params.number_of_folds)
    fold_number = 1

    trecs = []
    for train, test in folds:
        train_file = "features" + str(fold_number)
        test_file = "features_test" + str(fold_number)
        p = Pool(10)
        scores = p.map(partial(f, train_file, test_file, fold_number), C_array)
        # model_file = learn_svm(C, train_file, fold_number)
        # score_file = run_svm(C, model_file, test_file, fold_number)
        for score_file in scores:
            results = retrieve_scores(test, score_file)
            C = score_file.split("/")[2]
            trec = evaluator.create_trec_eval_file(test, queries, results, C)
            trecs.append(trec)
        fold_number += 1
    for trec in set(trecs):
        final = evaluator.order_trec_file(trec)
        C = final.split("_")[2]
        evaluator.run_trec_eval_on_test_correlation_svm(final, C)
