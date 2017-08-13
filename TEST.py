import preprocess as p
import models_handler as mh
import SVM_SGD as svmsgd
import evaluator as e
import params
if __name__=="__main__":
    a = p.preprocess("../../../svm_test/a/test.txt")
    X,y,queries=a.retrieve_data_from_file("../svm_test/a/test.txt")
    number_of_folds = 5
    number_of_queries = len(set(queries))
    eval = e.eval()
    eval.remove_score_file_from_last_run()
    if not params.recovery:
        eval.create_qrels_file(X,y,queries)
    folds = a.create_folds(X,y,queries,number_of_folds)
    fold_number = 1
    C_array = [0.1,0.01,0.001]
    model_handler = mh.models_handler(C_array)
    models={}
    validated = set()
    for train,test in folds:
        eval.empty_validation_files()
        validated, validation_set, train_set = a.create_validation_set(number_of_folds, validated, set(train),
                                                                       number_of_queries, queries)
        train_set = list(train_set)
        X_i, y_i = a.create_data_set(X[train_set], y[train_set], queries[train_set])
        model_handler.set_queries_to_folds(queries,test,fold_number)
        model_handler.fit_model_on_train_set_and_choose_best(X,X_i,y_i,validation_set,fold_number,queries)
        model_handler.predict(X,y,test,fold_number)
        fold_number+=1
    eval.run_trec_eval_on_test()




