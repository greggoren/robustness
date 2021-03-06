from bias_variance_tradeoff_svm import params
import numpy as np
import os
import subprocess
from functools import partial
from multiprocessing import Pool
import random
import pickle
def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    return iter(p.stdout.readline, b'')


def learn_svm(train_file, C):
    if not os.path.exists("./models_light/"):
        os.makedirs("./models_light/")
    model_file = "./models_light/svm_model" + str(C)
    learning_command = "./svm_rank_learn -c " + str(C) + " " + train_file + " " + model_file
    for output_line in run_command(learning_command):
        print(output_line)
    return model_file


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


def upload_models(models_dir):
    models = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            model = float(file.split("svm_model")[1])
            models.append(model)
    return models


def init():
    models = np.random.uniform(0, 10000, 60)
    return list(set(models))


if __name__ == "__main__":
    # C_array = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
    train_file = params.data_set_file
    random.seed(9001)
    # for C in C_array:
    #     model_file = learn_svm(C, train_file)
    #
    # C_array = [float(i) / 100 for i in range(1, 10)]
    # for C in C_array:
    #     model_file = learn_svm(C, train_file)
    #
    # C_array = [float(i) / 10 for i in range(1, 10)]
    # for C in C_array:
    #     model_file = learn_svm(C, train_file)
    #
    # C_array = [float(i) for i in range(1, 10)]
    # for C in C_array:
    #     model_file = learn_svm(C, train_file)
    #
    # C_array = [float(i * 100) for i in range(1, 10)]
    # for C in C_array:
    #     model_file = learn_svm(C, train_file)
    #
    # C_array = [float(i * 1000) for i in range(1, 10)]
    # C_array = [0.0001, 0.001, 0.01, 0.1]
    C_array = init()
    f = open("C_array", 'wb')
    pickle.dump(C_array, f)
    f.close()
    existing = upload_models("models_light")
    C_array = list(set(C_array) - set(existing))
    f = partial(learn_svm, train_file)
    with Pool(processes=10) as pool:
        pool.map(f, C_array)
