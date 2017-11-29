from coodinate_ascent import preprocess_clueweb as p
from coodinate_ascent import single_model_handler_coordinate_ascent as mh
from coodinate_ascent import evaluator as e
from coodinate_ascent import params
import sys
if __name__=="__main__":
    preprocess = p.preprocess()
    X,y,queries=preprocess.retrieve_data_from_file(params.data_set_file,params.normalized)
    sys.stdout.flush()
    number_of_queries = len(set(queries))
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict()
    evaluator.remove_score_file_from_last_run()
    sys.stdout.flush()
    train,validation = preprocess.create_test_train_split_cluweb(queries)
    sys.stdout.flush()
    train_file=preprocess.create_train_file(X[train], y[train], queries[train])
    test_file = preprocess.create_train_file(X[validation], y[validation], queries[validation],True)
    sys.stdout.flush()
    regularization_array = [0.1,0.01,0.001]
    single_model_handler = mh.single_model_handler_coordinate_ascent(regularization_array)

    single_model_handler.fit_model_on_train_set_and_choose_best_for_competition(train_file,test_file,validation,queries,evaluator)
    print("learning is finished")





