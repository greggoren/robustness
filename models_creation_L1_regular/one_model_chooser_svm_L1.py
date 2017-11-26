from models_creation_L1_regular import preprocess_clueweb as p
from models_creation_L1_regular import single_model_handler_svm_L1 as mh
from models_creation_L1_regular import evaluator_L1 as e
from models_creation_L1_regular import params_L1
import sys
import random as r

if __name__=="__main__":
    r.seed(params_L1.random_seed)  # traceability reasons
    labda = sys.argv[1]
    if labda=="N":
        labda=None
    else:
        labda = float(labda)
    C_array=[0.1,0.01,0.001,0.0001]
    preprocess = p.preprocess()
    score_file = params_L1.score_file
    X,y,queries=preprocess.retrieve_data_from_file(params_L1.data_set_file,params_L1.normalized)
    sys.stdout.flush()
    number_of_queries = len(set(queries))
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict()
    evaluator.remove_score_file_from_last_run(score_file)
    sys.stdout.flush()
    train,validation = preprocess.create_test_train_split_cluweb(queries)
    sys.stdout.flush()
    X_i,y_i=preprocess.create_data_set(X[train], y[train], queries[train])
    sys.stdout.flush()
    Lambda_array = [labda]
    single_model_handler = mh.single_model_handler_svm_L1(Lambda_array,C_array)
    single_model_handler.fit_model_on_train_set_and_choose_best_for_competition(X,y,X_i,y_i,validation,queries,evaluator,preprocess,score_file)
    print("learning is finished")





