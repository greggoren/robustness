import preprocess_clueweb as p
import svm_ent_models_handler as mh
import evaluator as e
import params
import sys
from multiprocessing import Pool
from functools import partial

def fit_models(X, y, svm):
    svm.fit(X, y)
    return svm

models = []

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
    Gamma_array = [0.2,0.1,0.01]

    model_handler = mh.svm_ent_models_handler(C_array,Gamma_array)
    models = model_handler.models
    validated = set()
    for train,test in folds:
        sys.stdout.flush()
        evaluator.empty_validation_files()
        validated, validation_set, train_set = preprocess.create_validation_set(params.number_of_folds, validated, set(train),
                                                                                number_of_queries, queries)
        X_i, y_i = preprocess.create_data_set(X[train_set], y[train_set], queries[train_set])
        f = partial(fit_models,X_i,y_i)
        p = Pool(4)
        new_models = p.map(f,models)
        model_handler.models = new_models
        model_handler.set_queries_to_folds(queries,test,fold_number)
        model_handler.fit_model_on_train_set_and_choose_best(X,X_i,y_i,validation_set,fold_number,queries,evaluator)
        model_handler.predict(X,queries,test,fold_number,evaluator)
        fold_number += 1
    evaluator.run_trec_eval_on_test()



