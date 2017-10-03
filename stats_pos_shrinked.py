import preprocess_clueweb as p
import svm_ent_models_handler_pos as mh
import evaluator_ent_pos as e
import params_ent_shrinked_pos as params_ent
import sys
import pickle
def fit_models(X, y, svm):
    svm.fit(X, y)
    return svm

if __name__=="__main__":
    gamma = float(sys.argv[1])
    sigma = float(sys.argv[2])
    preprocess = p.preprocess()
    X,y,queries=preprocess.retrieve_data_from_file(params_ent.data_set_file,params_ent.normalized)
    number_of_queries = len(set(queries))
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict()
    score_file = params_ent.score_file + str(gamma) + "_" + str(sigma)
    evaluator.remove_score_file_from_last_run(score_file)
    gammas = []
    sigmas = [sigma, ]
    gammas.append(gamma)

    c_array = [0.1, 0.01, 0.001]

    folds = preprocess.create_folds(X, y, queries, params_ent.number_of_folds)
    fold_number = 1
    model_handler = mh.svm_ent_models_handler_pos(c_array, gammas, sigmas)
    validated = set()
    for train,test in folds:
        sys.stdout.flush()
        validation_folder = params_ent.validation_folder+"/"+str(gamma)+"_"+str(sigma)
        evaluator.empty_validation_files(validation_folder)
        validated, validation_set, train_set = preprocess.create_validation_set(params_ent.number_of_folds, validated, set(train),
                                                                                number_of_queries, queries)
        X_i, y_i = preprocess.create_data_set(X[train_set], y[train_set], queries[train_set])
        model_handler.set_queries_to_folds(queries,test,fold_number)
        model_handler.fit_model_on_train_set_and_choose_best_opt(X, X_i, y_i, validation_set, fold_number, queries, score_file, evaluator)
        model_handler.predict_opt(X, queries, test, fold_number, score_file, evaluator)
        fold_number += 1
    summary_file = params_ent.summary_file+str(gamma)+"_"+str(sigma)
    evaluator.run_trec_eval_on_test_for_opt(score_file,summary_file)
    with open(params_ent.model_handler_file+str(gamma)+"_"+str(sigma), 'wb') as f:
        pickle.dump(model_handler, f, pickle.HIGHEST_PROTOCOL)


