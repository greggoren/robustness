from . import preprocess_clueweb as p
from . import single_model_handler_svm_entropy as mh
from . import evaluator_ent_pos_minmax as e
from . import params_ent_pos_minmax as params_ent
import sys
import random as r
if __name__=="__main__":
    r.seed(params_ent.random_seed)  # traceability reasons
    gamma = float(sys.argv[1])
    sigma = float(sys.argv[2])
    Sigma_array=[sigma]
    Gamma_array=[gamma]
    preprocess = p.preprocess()
    X,y,queries=preprocess.retrieve_data_from_file(params.data_set_file,params.normalized)
    sys.stdout.flush()
    number_of_queries = len(set(queries))
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict()
    evaluator.remove_score_file_from_last_run()
    sys.stdout.flush()
    train,validation = preprocess.create_test_train_split_cluweb(queries)
    sys.stdout.flush()
    X_i,y_i=preprocess.create_data_set(X[train], y[train], queries[train])
    sys.stdout.flush()
    C_array = [0.1,0.01,0.001]
    single_model_handler = mh.svm_sgd_entropy_pos_minmax(C_array, Gamma_array, Sigma_array)
    single_model_handler.fit_model_on_train_set_and_choose_best_for_competition(X,y,X_i,y_i,validation,queries,evaluator,preprocess)
    print("learning is finished")




