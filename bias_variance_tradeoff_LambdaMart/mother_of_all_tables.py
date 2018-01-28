import os
from bias_variance_tradeoff_LambdaMart import analyze as a
import prep as p
import numpy as np


def upload_models(models_dir):
    model_handlers = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            model_file = root + "/" + file
            model_handlers.append(model_file)
    return model_handlers


def get_banned(banned_file):
    banned_queries = {i: [] for i in [1, 2, 3, 4, 5]}
    with open(banned_file) as banned:
        for ban in banned:
            splitted = ban.split()
            banned_queries[int(splitted[0]) - 5].append(splitted[1])
    return banned_queries

if __name__ == "__main__":
    preprocess = p.preprocess()
    analyze = a.analyze()
    models = upload_models("models")
    # banned = get_banned("../banned2")
    banned = {i: [] for i in [1, 2, 3, 4, 5, 6, 7, 8]}
    # banned[2].append("164")
    # banned[2].append("010")
    # models = ["../model_250_50"]
    # competition_data = preprocess.extract_features_by_epoch("../features_asr_modified")
    competition_data = preprocess.extract_features_by_epoch("../featuresASR_combined")
    analyze.create_table(competition_data, models, banned)
