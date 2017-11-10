import preprocess_clueweb as p
import LambdaMart_models_handler as mh
import evaluator as e
import params
import pickle

def get_results(score_file,test_indices):
    results={}
    with open(score_file) as scores:
        for index,score in enumerate(scores):
            results[test_indices[index]]=score
    return results

if __name__=="__main__":
    preprocess = p.preprocess()
    X,y,queries=preprocess.retrieve_data_from_file(params.data_set_file,params.normalized)
    number_of_queries = len(set(queries))
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict()
    evaluator.remove_score_file_from_last_run()
    folds = preprocess.create_folds(X, y, queries, params.number_of_folds)
    fold_number = 1
    trees = [250,500]
    leaves=[5,10]
    model_handler = mh.models_handler(trees,leaves)
    validated = set()
    for train,test in folds:
        evaluator.empty_validation_files()
        validated, validation_set, train_set = preprocess.create_validation_set(params.number_of_folds, validated, set(train),
                                                                                number_of_queries, queries)

        model_handler.set_queries_to_folds(queries,test,fold_number)
        train_file = preprocess.create_train_file(X[train_set], y[train_set], queries[train_set])
        validation_file = preprocess.create_train_file(X[validation_set], y[validation_set], queries[validation_set], True)
        test_file = preprocess.create_train_file(X[test], y[test], queries[test], True)
        model_handler.fit_model_on_train_set_and_choose_best(train_file,validation_file,fold_number,params.qrels,evaluator)
        trees_number,leaf_number=model_handler.self.chosen_model_per_fold[fold_number]
        scores_file=model_handler.run_model(test_file,trees_number,leaf_number)
        results = get_results(scores_file,test)
        evaluator.create_trec_eval_file(test,queries,results,"_".join([str(a) for a in (trees_number,leaf_number)]))
        fold_number += 1
    evaluator.run_trec_eval_on_test()
    with open(params.model_handler_file,'wb') as f:
        pickle.dump(model_handler,f,pickle.HIGHEST_PROTOCOL)



