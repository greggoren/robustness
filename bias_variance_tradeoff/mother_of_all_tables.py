import os
from bias_variance_tradeoff import analyze as a
import prep as p
import numpy as np


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
    model_handlers = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            model_file = root + "/" + file
            w = recover_model(model_file)
            model_handlers.append((model_file, w))


if __name__ == "__main__":
    preprocess = p.preprocess()
    analyze = a.analysis()
    meta_mhs = []
    name_dict = {"pos_plus": "POS/NEG Max", "pos_plus_big": "POS/NEG Max", "pos_minus_big": "POS/NEG Min",
                 'pos_minus': "POS/NEG Min", 'squared_minus_big': "Squared Min", 'squared_plus_big': "Squared Max",
                 'squared_minus': "Squared Min", 'squared_plus': "Squared Max", "regular": "SVM",
                 "L1": "L1 regularization", "doubly": "Doubly regularization", "minmax": "Min Max", "maxmin": "Max Min",
                 "test": "L1"}
    reverse = {name_dict[a]: a for a in name_dict}
    dirs = ["test"]
    for dir in dirs:
        meta_mhs.append(create_mhs(dir))
    meta_model_objects = []
    for mhs in meta_mhs:
        meta_model_objects.append(preprocess.load_model_handlers(mhs))
    cd = preprocess.extract_features_by_epoch("features_asr_modified")
    analyze.create_table(meta_model_objects, cd, name_dict)
