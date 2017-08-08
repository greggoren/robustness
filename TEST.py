import preprocess as p
import numpy as np
if __name__=="__main__":
    a = p.preprocess("../../../svm_test/a/test.txt")
    X,y,queries=a.retrieve_data_from_file("../../../svm_test/a/test.txt")
    folds = a.create_folds(X,y,queries,5)


