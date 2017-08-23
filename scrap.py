import preprocess_clueweb as p
import svm_models_handler as mh
import evaluator as e
import params
import pickle
import numpy as np
import math
import time

def safe_ln( x):
    if x <= 0:
        return 0
    return math.log(x)

if __name__=="__main__":
    w = np.zeros(130)

    w+=1
    begin = time.time()
    print (begin)
    r_t = sum([(w_i ** 2) * safe_ln(w_i ** 2) for w_i in w])
    z_t = sum([(w_i ** 2) for w_i in w])
    print ("took",time.time()-begin)





