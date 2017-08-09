import preprocess as p
import numpy as np
import SVM_SGD as svmsgd
import evaluator as e
if __name__=="__main__":
    a = p.preprocess("../../../svm_test/a/test.txt")
    X,y,queries=a.retrieve_data_from_file("../../../svm_test/a/test.txt")
    folds = a.create_folds(X,y,queries,5)
    svm = svmsgd.svm_sgd(C=0.1)
    for train,test in folds:
        X_i,y_i=a.create_data_set(X[train],y[train],queries[train])
        X_i = np.matrix(X_i)
        svm.fit(X_i,y_i)
        svm.predict(X,queries,test)




