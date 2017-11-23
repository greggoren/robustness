from models_creation_doubly_regularized import preprocess_clueweb as p
from models_creation_doubly_regularized import single_model_handler_svm_doubly as mh
from models_creation_doubly_regularized import evaluator_doubly as e
from models_creation_doubly_regularized import params_doubly
import sys
import random as r

if __name__=="__main__":
    r.seed(params_doubly.random_seed)  # traceability reasons
    labda1 = sys.argv[1]
    labda2 = float(sys.argv[2])
    if labda1=="N":
        labda1=None
    else:
        labda1 = float(labda1)
    preprocess = p.preprocess()
    score_file = params_doubly.score_file
    X,y,queries=preprocess.retrieve_data_from_file(params_doubly.data_set_file,params_doubly.normalized)
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
    Lambda1_array = [labda1]
    Lambda2_array = [labda2]
    single_model_handler = mh.single_model_handler_svm_L1(Lambda1_array,Lambda2_array)
    single_model_handler.fit_model_on_train_set_and_choose_best_for_competition(X,y,X_i,y_i,validation,queries,evaluator,preprocess,score_file)
    print("learning is finished")





