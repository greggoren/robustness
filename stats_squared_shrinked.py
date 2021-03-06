import preprocess_clueweb as p
import svm_ent_models_handler as mh
import evaluator_ent as e
import params_ent as params_ent
import sys
import pickle
import random as r

if __name__=="__main__":
    r.seed(params_ent.random_seed)  # traceability reasons
    gamma = float(sys.argv[1])
    preprocess = p.preprocess()
    X,y,queries=preprocess.retrieve_data_from_file(params_ent.data_set_file,params_ent.normalized)
    number_of_queries = len(set(queries))
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict()
    score_file = params_ent.score_file + str(gamma)
    evaluator.remove_score_file_from_last_run(score_file)
    C_array = [0.1,0.01,0.001]
    folds = preprocess.create_folds(X, y, queries, params_ent.number_of_folds)
    fold_number = 1
    score_file = params_ent.score_file+str(gamma)
    Gamma_array = []
    Gamma_array.append(gamma)

    model_handler = mh.svm_ent_models_handler(C_array,Gamma_array)
    validated = set()
    for train,test in folds:
        sys.stdout.flush()
        validation_folder = params_ent.validation_folder + "/" + str(gamma)
        evaluator.empty_validation_files(validation_folder)
        validated, validation_set, train_set = preprocess.create_validation_set(params_ent.number_of_folds, validated, set(train),
                                                                                number_of_queries, queries)
        X_i, y_i = preprocess.create_data_set(X[train_set], y[train_set], queries[train_set])
        model_handler.set_queries_to_folds(queries,test,fold_number)
        model_handler.fit_model_on_train_set_and_choose_best_opt(X,X_i,y_i,validation_set,fold_number,queries,score_file,evaluator)
        model_handler.predict_opt(X,queries,test,fold_number,score_file,evaluator)
        fold_number += 1
    summary_file = params_ent.summary_file+str(gamma)
    evaluator.run_trec_eval_on_test_for_opt(score_file,summary_file)
    with open(params_ent.model_handler_file+str(gamma), 'wb') as f:
        pickle.dump(model_handler, f, pickle.HIGHEST_PROTOCOL)


