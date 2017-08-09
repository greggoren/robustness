import SVM_SGD as ss
import numpy as np
import preprocess as p
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
if __name__=="__main__":
    f = csr_matrix([[1,2],[2,2]])
    f= f.todense()
    print(f)

