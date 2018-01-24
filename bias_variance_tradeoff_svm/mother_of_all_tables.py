import os
from bias_variance_tradeoff_svm import analyze as a
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


def get_banned(banned_file):
    banned_queries = {i: [] for i in [1, 2, 3, 4, 5]}
    with open(banned_file) as banned:
        for ban in banned:
            splitted = ban.split()
            banned_queries[int(splitted[0])].append(splitted[1])
    return banned_queries
def upload_models(models_dir):
    model_handlers = {}
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            model_file = root + "/" + file
            w = recover_model(model_file)
            model_handlers[model_file] = w
    return model_handlers


if __name__ == "__main__":
    preprocess = p.preprocess()
    analyze = a.analyze()
    # svms = upload_models("models_light")
    banned = get_banned("../banned1")
    svms = {"svm_model0.1": pickle.load(open("../svm_model", 'rb'))}
    # competition_data = preprocess.extract_features_by_epoch("../features_asr_modified")
    competition_data = preprocess.extract_features_by_epoch("../featuresASR_round1_SVM")
    analyze.create_table(competition_data, svms, banned)
    # analyze.score_experiment(competition_data, svms)
