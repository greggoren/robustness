import SVM_SGD as svm_s
import numpy as np
import random as r
import math
from models_creation_doubly import params_L1
import sys
class svm_sgd_L1(svm_s.svm_sgd):
    def __init__(self, Lambda=None):
        self.Lambda = Lambda
        self.w = None



    def L1_norm_subgradient(self,number_of_features,w):
        subgradient = np.zeros(number_of_features)
        for i,w_i in enumerate(w):
            if w_i<0:
                subgradient[i]=-1
            else:
                subgradient[i]=1
        return subgradient


    def check_prediction(self,y_k,w):

        if np.dot(w,y_k.T)>1:
            return True
        return False

    def fit(self,X,y):
        print ("started SGD")
        Lambda=self.Lambda
        number_of_examples,number_of_features = len(X),len(X[0])
        if self.Lambda is None:
            Lambda = 1.0/math.sqrt(number_of_examples)
        w = np.zeros(number_of_features)#weights initialization
        self.w = np.zeros(number_of_features)#weights initialization
        iterations = params_L1.iter_factor * number_of_examples
        for t in range(iterations):#itarating over examples
            if t%1000000==0:
                print ("in iteration",t,"out of",iterations)
                sys.stdout.flush()
            lr = 1.0/(t+1)
            random_index = r.randint(0,number_of_examples-1)
            y_k = X[random_index]*y[random_index]
            if not self.check_prediction(y_k,w):
                w = self.w-lr*Lambda*self.L1_norm_subgradient(number_of_features,w) + lr*y_k
            else:
                w = self.w-lr*Lambda*self.L1_norm_subgradient(number_of_features,w)
            self.w=((self.w*t)+w)/(t+1)
        print ("SGD ended")


    def predict(self,X,queries,test_indices,eval,validation=None):
        results = {}
        for index in test_indices:
            results[index] = np.dot(self.w,X[index].T)
        return eval.create_trec_eval_file(test_indices,queries,results,str(self.Lambda),validation)

    def predict_opt(self,X,queries,test_indices,eval,score,validation=None):
        results = {}
        for index in test_indices:
            results[index] = np.dot(self.w,X[index].T)
        return eval.create_trec_eval_file_opt(test_indices,queries,results,str(self.Lambda),score,self.Lambda,validation)
