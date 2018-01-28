import os
from bias_variance_tradeoff_svm import analyze as a
import prep as p
import numpy as np
import pickle
import random
from copy import deepcopy
def create_mhs(dir):
    mhs = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            file_name = root + "/" + file
            model = file.split("pickle")[1]
            mhs.append((file_name, model, 'a'))
    return mhs


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


def get_banned(banned_file):
    banned_queries = {i: [] for i in [1, 2, 3, 4, 5]}
    with open(banned_file) as banned:
        for ban in banned:
            splitted = ban.split()
            banned_queries[int(splitted[0])].append(splitted[1])
    return banned_queries


def upload_models(models_dir, C_array):
    random.seed(9032)
    model_handlers = {}
    models = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            t = file.split("svm_model")[1]
            # if len(t.split(".")) > 1 and int(t.split(".")[0]) > 0 and int(t.split(".")[1]) > 0:
            model = float(file.split("svm_model")[1])

            model_file = root + "/" + file
            w = recover_model(model_file)
            model_handlers[model_file] = w
            models.append(model_file)
    # random.shuffle(models)
    # sampeled_models = models[:31]
    # res = deepcopy(model_handlers)
    # for i in model_handlers:
    #     if i in sampeled_models:
    #         res.pop(i)

    return model_handlers


if __name__ == "__main__":
    # C_array = pickle.load(open("C_array", 'rb'))
    # C_array = [1000, 2000, 3000, 4000, 5000]
    # C_array=[]
    # C_array.extend([(i + 1)/10000  for i in range(10)])
    # C_array.extend([(i + 1)/1000 for i in range(10)])
    # C_array.extend([(i + 1) / 100 for i in range(5)])
    # C_array.extend([(i + 1) / 10 for i in range(5)])
    # C_array.extend([(i + 1)  for i in range(10)])
    # C_array.extend([(i + 1)*100  for i in range(10)])
    # C_array.extend([(i + 1) *1000 for i in range(10)])
    # C_array.extend([(i + 1) for i in range(5)])
    # C_array.extend([(i + 1) * 10 for i in range(5)])
    # C_array.extend([(i + 1) * 100 for i in range(5)])
    # C_array.extend([900, 800, 600, 700])
    C_array = []
    svms = upload_models("models_light", C_array)
    preprocess = p.preprocess()
    analyze = a.analyze()

    # banned = get_banned("../banned2")
    banned = {i: [] for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    # banned[2].append("164")
    # svms = {"svm_model0.1": pickle.load(open("../svm_model", 'rb'))}
    competition_data = preprocess.extract_features_by_epoch("../features_asr_modified")
    # competition_data = preprocess.extract_features_by_epoch("../featuresASR_round2_SVM")
    analyze.create_table(competition_data, svms, banned)
    # analyze.score_experiment(competition_data, svms)
