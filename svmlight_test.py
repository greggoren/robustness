import preprocess_clueweb as p
import models_handler as mh
import evaluator as e
import params
from sklearn.datasets import dump_svmlight_file
import subprocess

def run_command(self, command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    return iter(p.stdout.readline, b'')

def learn_svm(C,train_file,fold):
    learning_command = "./svm_rank_learn -c " + str(C) + " "+train_file+" "+"models/"+str(fold)+"/svm_model"+str(C)+".txt"
    for output_line in run_command(learning_command):
        print(output_line)
    return "models/"+str(fold)+"/svm_model"+str(C)+".txt"


def recover_model(model):
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
                            model_wheights_per_fold[fold].append(0)
                            indexes_covered.append(repair)
                    model_wheights_per_fold[fold].append(float(wheights[index].split(":")[1]))
                    indexes_covered.append(feature_id)

if __name__=="__main__":
    preprocess = p.preprocess()
    X,y,queries=preprocess.retrieve_data_from_file(params.data_set_file)
    number_of_queries = len(set(queries))
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict()
    evaluator.remove_score_file_from_last_run()
    if not params.recovery:
        evaluator.create_qrels_file(X, y, queries)
    folds = preprocess.create_folds(X, y, queries, params.number_of_folds)
    fold_number = 1
    C_array = [0.1,0.01,0.001,1000,100]
    model_handler = mh.models_handler(C_array)
    validated = set()
    for train,test in folds:
        evaluator.empty_validation_files()
        validated, validation_set, train_set = preprocess.create_validation_set(params.number_of_folds, validated, set(train),
                                                                                number_of_queries, queries)
        #X_i, y_i = preprocess.create_data_set(X[train_set], y[train_set], queries[train_set])
        train_file = "train" + str(fold_number) + ".txt"
        dump_svmlight_file(X[train],y[train],train_file,query_id=queries[train])
        for C in C_array:
            model_file = learn_svm(C,train_file)

