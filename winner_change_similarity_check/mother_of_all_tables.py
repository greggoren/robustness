import os
from  winner_change_similarity_check import analyze_svm as a
import prep as p
import numpy as np
import pickle

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


def upload_models(models_dir):
    model_handlers = {}
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            model_file = root + "/" + file
            w = recover_model(model_file)
            model_handlers[model_file] = w
    return model_handlers


def get_banned(banned_file):
    banned_queries = {i: [] for i in [0, 1, 2, 3, 4, 5]}
    with open(banned_file) as banned:
        for ban in banned:
            splitted = ban.split()
            banned_queries[int(splitted[0])].append(splitted[1])
    return banned_queries


if __name__ == "__main__":
    preprocess = p.preprocess()
    analyze = a.analyze()
    model_file = open("../svm_model", 'rb')
    banned_queries = get_banned("../banned")
    w = pickle.load(model_file)

    models = {"svm": w}
    competition_data = preprocess.extract_features_by_epoch("../featuresASR_S")
    analyze.create_table(competition_data, models, banned_queries)
