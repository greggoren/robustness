from variance_exp_svm import preprocess_clueweb as p
from variance_exp_svm import params
import numpy as np
import os
import subprocess
from multiprocessing import Pool
from functools import partial
import time
import pickle


def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    return iter(p.stdout.readline, b'')


def learn_svm(C, train_file, fold, subset):
    if not os.path.exists("models/" + str(fold)):
        try:
            os.makedirs("models/" + str(fold))
        except:
            print("weird behaviour")

    learning_command = "./svm_rank_learn -c " + str(C) + " " + train_file + " " + "models/" + str(
        fold) + "/svm_model" + str(C) + str(subset)
    for output_line in run_command(learning_command):
        print(output_line)
    return "models/" + str(fold) + "/svm_model" + str(C) + str(subset)


def run_svm(C, model_file, test_file, fold, subset):
    score_path = "scores/" + str(fold)
    if not os.path.exists(score_path):
        try:
            os.makedirs(score_path)
        except:
            print("collition")
    rank_command = "./svm_rank_classify " + test_file + " " + model_file + " " + score_path + "/" + str(C) + "#" + str(
        subset)
    for output_line in run_command(rank_command):
        print(output_line)
    return score_path + "/" + str(C) + "#" + str(subset)


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


def f(fold, C, subset):
    train_file = "../variance_exp/train/" + str(fold) + "/features" + str(subset)
    test_file = "../variance_exp/test/" + str(fold) + "/features0_test"
    model_file = learn_svm(C, train_file, fold, subset)
    score_file = run_svm(C, model_file, test_file, fold, subset)
    return score_file


def update_scores(results, scores, subset, C):
    for index in results:
        scores[C][subset][index].append(results[index])
    return scores


if __name__ == "__main__":

    C_array = [float(i + 1) / 1000 for i in range(10)]
    C_array.extend([float(i + 1) / 100 for i in range(10)])
    C_array.extend([float(i + 1) / 10 for i in range(10)])
    C_array.extend([float(i + 1) for i in range(10)])
    C_array.extend([float(i + 1) * 10 for i in range(10)])
    # C_array.extend([float(i + 1) * 100 for i in range(10)])

    start = time.time()
    preprocess = p.preprocess()
    X, y, queries = preprocess.retrieve_data_from_file(params.data_set_file, params.normalized)
    scores = {C: {subset: {i: [] for i in range(len(queries))} for subset in range(31)} for C in C_array}
    number_of_queries = len(set(queries))

    for C in C_array:
        fold_number = 1
        folds = preprocess.create_folds(X, y, queries, 5)
        for train, test in folds:
            p = Pool(10)
            func = partial(f, fold_number, C)
            score_files = p.map(func, range(31))
            for score in score_files:
                subset = int(score.split("#")[1])
                results = retrieve_scores(test, score)
                scores = update_scores(results, scores, subset, C)
            fold_number += 1
    print("it took:", time.time() - start)
    with open("variance_data", 'wb') as data:
        pickle.dump(scores, data)
