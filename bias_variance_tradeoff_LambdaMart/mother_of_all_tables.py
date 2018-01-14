import os
from bias_variance_tradeoff_LambdaMart import analyze_query_level as a
import prep as p
import numpy as np


def upload_models(models_dir):
    model_handlers = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            model_file = root + "/" + file
            model_handlers.append(model_file)
    return model_handlers


if __name__ == "__main__":
    preprocess = p.preprocess()
    analyze = a.analyze()
    models = upload_models("models")
    competition_data = preprocess.extract_features_by_epoch("../features_asr_modified")
    analyze.create_table(competition_data, models, [])
