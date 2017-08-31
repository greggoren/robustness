import preprocess_clueweb as p
import svm_ent_models_handler as mh
import evaluator_ent as e
import numpy as np
import params_ent
import pickle
def fit_models(X, y, svm):
    svm.fit(X, y)
    return svm


def set_qid_for_trec(query):
    if query < 10:
        qid = "00" + str(query)
    elif query < 100:
        qid = "0" + str(query)
    else:
        qid = str(query)
    return qid


if __name__=="__main__":

    preprocess = p.preprocess()
    X,y,queries=preprocess.retrieve_data_from_file("features",False)
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict()
    file = open("model_handler_ent_opt.pickle",'rb')
    mh = pickle.load(file)
    with open("test_res.txt",'w') as f:
        for index in range(len(X)):
            q = queries[index]
            fold = mh.query_to_fold_index[q]
            weights= mh.weights_index[fold]
            f.write(set_qid_for_trec(queries[index])+" Q0 "+evaluator.doc_name_index[index]+" "+str(0)+" "+str(np.dot(X[index],weights.T))+" seo\n")



