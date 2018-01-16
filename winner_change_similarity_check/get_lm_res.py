import os
from  winner_change_similarity_check import analyze_lm as a
import prep as p
import numpy as np
import pickle


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

    models = {"../testmodel_250_50": "../testmodel250_50"}
    competition_data = preprocess.extract_features_by_epoch("../featuresASR_L")
    analyze.create_table(competition_data, models, banned_queries)
