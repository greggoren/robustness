import preprocess_clueweb as p
import svm_models_handler as mh
import evaluator as e
import params
import pickle
if __name__=="__main__":
    preprocess = p.preprocess()
    X,y,queries=preprocess.retrieve_data_from_file(params.data_set_file,params.normalized)
    number_of_queries = len(set(queries))
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict()
    evaluator.remove_score_file_from_last_run()
    """if not params.recovery:
        evaluator.create_qrels_file(X, y, queries)"""
    folds = preprocess.create_folds(X, y, queries, params.number_of_folds)
    fold_number = 1
    C_array = [0.1,0.01,0.001]
    model_handler = mh.models_handler(C_array)
    validated = set()
    for train,test in folds:
        print("queries in test:",set(queries[test]))
        evaluator.empty_validation_files()
        validated, validation_set, train_set = preprocess.create_validation_set(params.number_of_folds, validated, set(train),
                                                                                number_of_queries, queries)
        X_i, y_i = preprocess.create_data_set(X[train_set], y[train_set], queries[train_set])
        #X_i, y_i = preprocess.create_data_set_opt(X[train_set], y[train_set], queries[train_set])
        model_handler.set_queries_to_folds(queries,test,fold_number)
        model_handler.fit_model_on_train_set_and_choose_best(X,X_i,y_i,validation_set,fold_number,queries,evaluator)
        model_handler.predict(X,queries,test,fold_number,evaluator)
        fold_number += 1
    evaluator.run_trec_eval_on_test()
    with open(params.model_handler_file,'wb') as f:
        pickle.dump(model_handler,f,pickle.HIGHEST_PROTOCOL)



