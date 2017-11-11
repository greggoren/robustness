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
    model_handler = mh.model_handler_LambdaMart(trees,leaves)
    validated = set()
    for train,test in folds:
        evaluator.empty_validation_files()
        validated, validation_set, train_set = preprocess.create_validation_set(params.number_of_folds, validated, set(train),
                                                                                number_of_queries, queries)
        validation_set=list(validation_set)
        model_handler.set_queries_to_folds(queries,test,fold_number)
        train_file = preprocess.create_train_file(X[train_set], y[train_set], queries[train_set])
        validation_file = preprocess.create_train_file(X[list(validation_set)], y[list(validation_set)], queries[list(validation_set)], True)
        model_handler.fit_model_on_train_set_and_choose_best(train_file,validation_file,validation_set,queries,fold_number,params.qrels,evaluator)
        trees_number,leaf_number=model_handler.chosen_model_per_fold[fold_number]
        test_file = preprocess.create_train_file_cv(X[test], y[test], queries[test],fold_number, True)
        scores_file=model_handler.run_model_on_test(test_file,fold_number,trees_number,leaf_number)
        results = model_handler.retrieve_scores(test,scores_file)
        evaluator.create_trec_eval_file(test,queries,results,"_".join([str(a) for a in (trees_number,leaf_number)]))
        # final_trec_eval = evaluator.order_trec_file(trec_file)
        fold_number += 1
    final=evaluator.order_trec_file(params.score_file)
    evaluator.run_trec_eval_on_test(final)




