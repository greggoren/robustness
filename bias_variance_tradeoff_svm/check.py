import numpy as np
import os
from numpy.linalg import norm
from scipy.stats import pearsonr
from scipy.stats import spearmanr


def upload_models(models_dir):
    model_handlers = {}
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            model_file = root + "/" + file
            w = recover_model(model_file)
            model_handlers[model_file] = w
    return model_handlers


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


svms = upload_models("models_light")
C = []
Norm = []
for svm in svms:
    C.append(float(svm.split("svm_model")[1]))
    Norm.append(norm(svms[svm]))

print(len(C))
print(norm(svms['models_light/svm_model0.0001']))
print(pearsonr(C, Norm))
