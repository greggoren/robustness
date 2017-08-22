import preprocess_clueweb as p
import models_handler as mh
import evaluator as e
import params
from sklearn.datasets import dump_svmlight_file
import pickle
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
        dump_svmlight_file(X[train],y[train],"train"+str(fold_number)+".txt",query_id=queries[train])
        for C in C_array:
            "train models with svmlight"
