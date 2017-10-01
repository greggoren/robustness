import numpy as np
import math
import prep as p
def entropy_part_for_sgd(number_of_features,w):
    r_t, z_t = 0, 0
    for i in w:
        r_t += (i ** 2) * safe_ln(i ** 2)
        z_t += i ** 2

    addition = np.zeros(number_of_features)
    if z_t != 0:  # avoid division by zero
        for i, w_i in enumerate(w):
            print(w_i * ((safe_ln(w_i ** 2)) / z_t - r_t / (z_t ** 2)))
            print(safe_ln(w_i ** 2)/z_t)
            addition[i] = w_i * ((safe_ln(w_i ** 2)) / z_t - r_t / (z_t ** 2))
    return addition

def entropy_part_for_sgd_sq(w):
    z_t = 0
    mes = 0
    for i in w:
        z_t += (i**2)
    for i in w:
        mes-= (((i ** 2)/z_t) * safe_ln((i ** 2)/z_t))
    return mes

def entropy_part_for_sgd_pos(w):
        r_t_pos,r_t_neg = 0, 0
        for i in w:
            if i<0:
                r_t_neg += (-i) * safe_ln(-i)
            else:
                r_t_pos += (i) * safe_ln(i)
        return r_t_pos,r_t_neg
def safe_ln(x):
    if x <= 0:
        return 0
    return math.log(x,2)

if __name__=="__main__":
    # mhs = [("../model_handler_pos_minus.pickle0.001", '0.001', 'b'), ("../model_handler_pos_minus.pickle0.01", '0.01', 'g'),
    #        ("../model_handler_pos_minus.pickle0.1", '0.1', 'r'), ("../model_handler_pos_minus.pickle0.2", '0.2', 'm'),
    #        ("../data/model_handler_asr_cmp.pickle", 'svm', 'k')]
    # mhs = [("model_handler_ent_opt_shrinked_minus.pickle0.001", '0.001', 'b'),
    #        ("model_handler_ent_opt_shrinked_minus.pickle0.01", '0.01', 'g'),
    #        ("model_handler_ent_opt_shrinked_minus.pickle0.1", '0.1', 'r'),
    #        ("model_handler_ent_opt_shrinked_minus.pickle0.2", '0.2', 'm'),
    #        ("data/model_handler_asr_cmp.pickle", 'svm', 'k')]
    mhs = [("model_handler_ent_opt_shrinked1.pickle0.001", '0.001', 'b'),
           ("model_handler_ent_opt_shrinked1.pickle0.01", '0.01', 'g'),
           ("model_handler_ent_opt_shrinked1.pickle0.1", '0.1', 'r'),
           ("model_handler_ent_opt_shrinked1.pickle0.2", '0.2', 'm'), ("data/model_handler_asr_cmp.pickle", 'svm', 'k')]
    # mhs = [("../model_handler_ent_opt_shrinked_minus.pickle0.001", '0.001', 'b'),
    #        ("../model_handler_ent_opt_shrinked_minus.pickle0.01", '0.01', 'g'),
    #        ("../model_handler_ent_opt_shrinked_minus.pickle0.1", '0.1', 'r'),
    #        ("../model_handler_ent_opt_shrinked_minus.pickle0.2", '0.2', 'm'),
    #        ("../data/model_handler_asr_cmp.pickle", 'svm', 'k')]
    # mhs = [("../model_handler_ent_opt_minus.pickle0.001", '0.001', 'b'),
    #        ("../model_handler_ent_opt_minus.pickle0.01", '0.01', 'g'),
    #        ("../model_handler_ent_opt_minus.pickle0.1", '0.1', 'r'), ("../model_handler_ent_opt_minus.pickle0.2", '0.2', 'm'),
    #        ("../data/model_handler_asr_cmp.pickle", 'svm', 'k')]
    preprocess = p.preprocess()
    mh_svm = preprocess.load_model_handlers(mhs)
    for svm in mh_svm:
        print(svm[2])
        print(entropy_part_for_sgd_sq(svm[0].weights_index[3]))
        print (svm[0].chosen_model_per_fold[3])

