import preprocess_clueweb as p
import models_handler as mh
import evaluator as e
import params
import pickle
if __name__=="__main__":
    preprocess = p.preprocess()
    X,y,queries=preprocess.retrieve_data_from_file(params.data_set_file)
    number_of_queries = len(set(queries))
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict()

    folds = preprocess.create_folds(X, y, queries, params.number_of_folds)
    validated = set()
    for train,test in folds:
        print("queries in test:",sorted(list(set(queries[test]))))
        validated, validation_set, train_set = preprocess.create_validation_set(params.number_of_folds, validated,
                                                                                set(train),
                                                                                number_of_queries, queries)
        print("queries in val:",sorted(list(set(queries[validation_set]))))



