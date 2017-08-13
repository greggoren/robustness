import preprocess as p
import numpy as np
import SVM_SGD as svmsgd
import evaluator as e
import params
if __name__=="__main__":
    a = p.preprocess("../../../svm_test/a/test.txt")
    X,y,queries=a.retrieve_data_from_file("../svm_test/a/test.txt")
    number_of_folds = 5
    number_of_queries = len(set(queries))
    eval = e.eval()
    eval.create_qrels_file(X,y,queries)
    folds = a.create_folds(X,y,queries,number_of_folds)
    svm = svmsgd.svm_sgd(C=0.1)

    models={}
    validated = set()
    for train,test in folds:
        eval.empty_validation_files()
        validation_results={}
        validated,validation_set,train_set = a.create_validation_set(number_of_folds,validated,set(train),number_of_queries,queries)
        train_set = list(train_set)
        X_i,y_i=a.create_data_set(X[train_set],y[train_set],queries[train_set])
        svm.fit(X_i,y_i)
        models[svm.C]=svm.w
        score_file = svm.predict(X,queries,validation_set,True)
        score=eval.run_trec_eval(params.qrels, score_file)
        validation_results[svm.C]=score
        print("validation score",score)
        #after getting argmax
        score_file=svm.predict(X,queries,test)
    eval.run_trec_eval_on_test(params.qrels,score_file,str(svm.C))




