import preprocess_clueweb as p
import single_model_handler as mh
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
    train,validation = preprocess.create_test_train_split_cluweb(queries)
    X_i,y_i=preprocess.create_data_set(X[train], y[train], queries[train])
    C_array = [0.1,0.01,0.001]
    single_model_handler = mh.single_model_handler(C_array)
    single_model_handler.fit_model_on_train_set_and_choose_best_for_competition(X,y,X_i,y_i,validation,queries,evaluator,preprocess)
    print("learning is finished")





