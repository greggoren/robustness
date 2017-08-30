import random as r
import params_ent
import preprocess_clueweb as p
from sklearn.model_selection import GroupKFold
from sklearn.datasets import dump_svmlight_file
import numpy as np
if __name__=="__main__":
    preprocess = p.preprocess()
    X, y, queries = preprocess.retrieve_data_from_file("featuresCB_asr",True)
    print(X.max(axis=0))
    print(X.min(axis=0))
    # dump_svmlight_file(X,y,"nan",zero_based=False,query_id=queries)


