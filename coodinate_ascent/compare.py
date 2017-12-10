import math
import numpy as np
import pickle
def get_L1_norm(w):
    norm = 0
    for w_i in w:
        norm+=abs(w_i)

    return norm

def get_L2_norm(w):
    norm=0
    for w_i in w:
        norm+=w_i**2
    return math.sqrt(norm)

def get_poitives_entropy(w):
    z_t_pos  =  0
    for i in w:
        if i>0:
            z_t_pos += i
    ent = 0
    for w_i in w:
        if w_i > 0:
            p=float(w_i)/z_t_pos
            ent+=p*safe_ln(p)
    return ent


def get_negatives_entropy(w):
    z_t_neg  =  0
    for i in w:
        if i<0:
            z_t_neg += -i
    ent = 0
    for w_i in w:
        if w_i < 0:
            p=float(-w_i)/z_t_neg
            ent+=p*safe_ln(p)
    return ent

def safe_ln(x):
    if x <= 0:
        return 0
    return math.log(x)


def recover_model(model):
    indexes_covered = []
    weights =[]
    with open(model) as model_file:
        for line in model_file:
            if line.__contains__(":"):
                wheights = line.split()
                for w in wheights:
                    weights.append(float(w.split(":")[1]))

    return np.array(weights)

w_ca= recover_model("testmodel_0.01")
print(w_ca)
w_svm=pickle.load(open("svm_model",'rb'))
print(w_svm)
print("svm")
print("L1:",get_L1_norm(w_svm))
print("L2:",get_L2_norm(w_svm))
print("positive entropy:",get_poitives_entropy(w_svm))
print("negative entropy:",get_negatives_entropy(w_svm))
print("coordinate ascent")
print("L1:",get_L1_norm(w_ca))
print("L2:",get_L2_norm(w_ca))
print("positive entropy:",get_poitives_entropy(w_ca))
print("negative entropy:",get_negatives_entropy(w_ca))