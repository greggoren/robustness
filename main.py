import random as r
import params_ent
import preprocess_clueweb as p
from sklearn.model_selection import GroupKFold
import numpy as np
if __name__=="__main__":
    preprocess = p.preprocess()
    X, y, queries = preprocess.retrieve_data_from_file("featuresCB_asr")
    print(np.where(np.isnan(X)))


