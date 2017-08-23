import preprocess_clueweb as p
import svm_models_handler as mh
import operator

import evaluator as e
import params
import numpy as np
import os
from sklearn.datasets import dump_svmlight_file
import subprocess
import SVM_SGD as s

def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    return iter(p.stdout.readline, b'')

def learn_svm(C,train_file,fold):
    if not os.path.exists("models/"+str(fold)):
        os.makedirs("models/"+str(fold))
    learning_command = "./svm_rank_learn -c " + str(C) + " "+train_file+" "+"models/"+str(fold)+"/svm_model"+str(C)+".txt"
    for output_line in run_command(learning_command):
        print(output_line)
    return "models/"+str(fold)+"/svm_model"+str(C)+".txt"


def recover_model(model):
    indexes_covered = []
    weights =[]
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

if __name__=="__main__":
    preprocess = p.preprocess()
    X,y,queries=preprocess.retrieve_data_from_file(params.data_set_file)
    number_of_queries = len(set(queries))
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict()
    evaluator.remove_score_file_from_last_run()
    folds = preprocess.create_folds(X, y, queries, params.number_of_folds)
    fold_number = 1
    C_array = [0.1,0.01,0.001]
    model_handler = mh.models_handler(C_array)
    validated = set()
    scores = {}
    models = {}
    for train,test in folds:
        print("queries in test:", set(queries[test]))
        evaluator.empty_validation_files()
        validated, validation_set, train_set = preprocess.create_validation_set(params.number_of_folds, validated, set(train),
                                                                                number_of_queries, queries)
        train_file = "train" + str(fold_number) + ".txt"
        run_command("rm "+train_file)
        dump_svmlight_file(X[train],y[train],train_file,query_id=queries[train],zero_based=False)
        for C in C_array:
            model_file = learn_svm(C,train_file,fold_number)
            weights = recover_model(model_file)
            svm = s.svm_sgd(C)
            svm.w = weights
            score_file=svm.predict(X, queries, validation_set,evaluator, True)
            score = evaluator.run_trec_eval(score_file)
            scores[svm.C] = score
            models[svm.C] = svm
        max_C = max(scores.items(), key=operator.itemgetter(1))[0]
        chosen_model = models[max_C]
        chosen_model.predict(X,queries,test,evaluator)
        fold_number+=1
    evaluator.run_trec_eval_on_test()



