import SVM_SGD as ss
import numpy as np
import preprocess as p
from sklearn.decomposition import PCA

if __name__=="__main__":
    data_set_location = "../../../svm_test/a/test.txt"
    """prep = p.preprocess(data_set_location)
    a,b=prep.index_features_for_competitors(True)
    X,y = prep.create_data_set_svm_rank(a,b)"""
    prep = p.preprocess(data_set_location)
    a,b,c=prep.retrieve_data_from_file(data_set_location)
    X,y = prep.create_data_set(a,b,c)
    svm = ss.svm_sgd(C=0.1)
    svm.fit(X,np.array(y))
    print svm.w

