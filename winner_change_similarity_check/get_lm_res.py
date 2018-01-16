import os
from  winner_change_similarity_check import analyze as a
import prep as p
import numpy as np
import pickle

if __name__ == "__main__":
    preprocess = p.preprocess()
    analyze = a.analyze()
    model_file = open("../svm_model", 'rb')
    banned_queries = []  # get_banned("banned")

    models = {"../testmodel_250_50": "../testmodel250_50"}
    competition_data = preprocess.extract_features_by_epoch("../features_asr_modified")
    analyze.create_table(competition_data, models, [])
