import numpy as np
import math
import time

def safe_ln(x):
    if x == 0:
        return 0
    return math.log(x)

def t(w,number_of_features):
    r_t, z_t = 0, 0
    for i in w:
        r_t += (i ** 2) * safe_ln(i**2)
        z_t += i ** 2
    addition = np.zeros(number_of_features)
    if z_t != 0:  # avoid division by zero
        for i, w_i in enumerate(w):
            addition[i] = w_i * ((safe_ln(w_i ** 2)) / z_t - r_t / (z_t ** 2))
    return addition

def entropy_part_for_sgd(w,number_of_features):
    """r_t = sum([(w_i ** 2) * safe_ln(w_i ** 2) for w_i in w])
    z_t = sum([w_i ** 2 for w_i in w])"""
    r_t,z_t = 0,0
    for i in w:
        r_t+=(i**2)*safe_ln(i)
        z_t += i**2
    addition = []
    if z_t != 0:  # avoid division by zero
        for w_i in w:
            addition.append(w_i * ((safe_ln(w_i ** 2)) / z_t - r_t / (z_t ** 2)))
    return np.array(addition)



if __name__=="__main__":
    w = np.zeros(130)
    w+=1
    begin = time.time()
    print (begin)
    for i in range(3000):
        w += t(w,130)
    print ("took",time.time()-begin)



