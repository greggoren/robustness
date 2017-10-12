import numpy as np
import math
from scipy.stats import kendalltau
import RBO as r
def safe_ln(x):
    if x <= 0:
        return 0
    return math.log(x)

def entropy_part_for_sgd(number_of_features,w):
    r_t, z_t = 0, 0
    for i in w:
        r_t += (i ** 2) * safe_ln(i ** 2)
        z_t += i ** 2
    addition = np.zeros(number_of_features)
    if z_t != 0:  # avoid division by zero
        for i, w_i in enumerate(w):
            print (w_i)
            addition[i] = w_i * ((safe_ln(w_i ** 2)) / z_t - r_t / (z_t ** 2))
    return addition
if __name__=="__main__":

    w = [1,2,3,4,5]
    a = [1, 2, 4, 3, 5]
    w_t = [2,1,3,4,5]
    a_t = [1, 2, 3, 4, 5]

    w1=[len(w)-d+1 for d in w]
    d1 = {w1.index(j): j for j in w1}
    w1_t = [len(w_t) - e + 1 for e in w_t]


    a1 = [len(a) - e + 1 for e in a]
    a1_t = [len(a_t) - e + 1 for e in a_t]
    d2 = {a1.index(j): j for j in a1}
    d2_t = {a1_t.index(j): j for j in a1_t}
    d1_t = {w1_t.index(j): j for j in w1_t}

    print (kendalltau(a1,w1)[0])
    print (kendalltau(a1_t,w1_t)[0])
    print(r.rbo_dict(d1,d2,0.95))
    print(r.rbo_dict(d1_t,d2_t,0.95))
