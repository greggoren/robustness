import SVM_SGD as svm_s
import numpy as np
import random as r
import math
from models_creation_minmax import params_ent_pos_minmax as params_ent
import sys
class svm_sgd_entropy_pos_minmax(svm_s.svm_sgd):
    def __init__(self, C=None,Gamma =None,Sigma=None):
        self.C = C
        self.w = None
        self.Gamma = Gamma
        self.Sigma = Sigma

    def entropy_part_for_sgd(self,number_of_features):
        r_t_pos, z_t_pos,z_t_neg,r_t_neg = 0,0,0,0
        for i in self.w:
            if i<0:
                r_t_neg += (-i) * self.safe_ln(-i)
                z_t_neg += -i
            else:
                r_t_pos += (i) * self.safe_ln(i)
                z_t_pos += i
        addition_pos= np.zeros(number_of_features)
        addition_neg = np.zeros(number_of_features)

        for i,w_i in enumerate(self.w):
            if w_i<0:
                addition_neg[i] = float((-self.safe_ln(-w_i)) / z_t_neg) - (float(r_t_neg) / (z_t_neg ** 2))
            else:
                if z_t_pos>0:
                    addition_pos[i] = (float(self.safe_ln(w_i)) / z_t_pos) + (float(r_t_pos)/ (z_t_pos ** 2))
        return self.Gamma*addition_pos+self.Sigma*addition_neg


    def safe_ln(self,x):
        if x <= 0:
            return 0
        return math.log(x)

    def fit(self,X,y):

        print ("started SGD")
        number_of_examples,number_of_features = len(X),len(X[0])
        self.w = np.zeros(number_of_features)#weights initialization
        tmp = list(range(number_of_examples))
        # r.shuffle(tmp)
        validation = tmp[:100000]
        if self.C is not None:
            lambda_factor = self.C*number_of_examples
        else:
            lambda_factor = number_of_examples
            self.C = 0
        iterations = params_ent.iter_factor * number_of_examples
        for t in range(iterations):#itarating over examples
            if t%1000000==0:
                print ("in iteration",t,"out of",iterations)
                sys.stdout.flush()
            if t%50000==0:
                error_rate = self.check_validation(validation, y, X)
                print("error_rate is ", error_rate)
                sys.stdout.flush()
            lr = 1.0/(t+2)
            random_index = r.randint(0,number_of_examples-1)
            y_k = X[random_index]*y[random_index]
            if not self.check_prediction(y_k):
                self.w = t*lr*self.w + lr*lambda_factor*y_k+self.entropy_part_for_sgd(number_of_features)*lr
            else:
                self.w = t * lr * self.w + self.entropy_part_for_sgd(number_of_features)*lr
        print ("SGD ended")


    def predict(self,X,queries,test_indices,eval,validation=None):
        results = {}
        for index in test_indices:
            results[index] = np.dot(self.w,X[index].T)
        return eval.create_trec_eval_file(test_indices,queries,results,str(self.C)+"_"+str(self.Gamma)+"_"+str(self.Sigma),validation)

    def predict_opt(self,X,queries,test_indices,eval,score,validation=None):
        results = {}
        for index in test_indices:
            results[index] = np.dot(self.w,X[index].T)
        return eval.create_trec_eval_file_opt(test_indices,queries,results,str(self.C)+"_"+str(self.Gamma)+"_"+str(self.Sigma),score,self.Gamma,self.Sigma,validation)
