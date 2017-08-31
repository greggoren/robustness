import random as r
import params_ent
import preprocess_clueweb as p
from sklearn.model_selection import GroupKFold
from sklearn.datasets import dump_svmlight_file
import numpy as np
if __name__=="__main__":
    preprocess = p.preprocess()
    X, y, queries = preprocess.retrieve_data_from_file("features",True)
    # dump_svmlight_file(X,y,"nan",zero_based=False,query_id=queries)
    xmx = X.max(axis=0)
    xmn= X.min(axis=0)
    for i in range (len(xmn)):
        if xmx[i]<=xmn[i]:
            print("BIG SHIT!!!!")



