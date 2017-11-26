import SVM_SGD as svm_s
import numpy as np
import random as r
import math
from models_creation_doubly_regularized_C import params_doubly
import sys
class svm_sgd_doubly(svm_s.svm_sgd):
    def __init__(self, Lambda1=None,Lambda2=None,C=None):
        self.Lambda1 = Lambda1
        self.Lambda2 = Lambda2
        self.C=C
        self.w = None



    def L1_norm_subgradient(self,number_of_features):
        subgradient = np.zeros(number_of_features)
        for i,w_i in enumerate(self.w):
            if w_i<0:
                subgradient[i]=-1
            else:
                subgradient[i]=1
        return subgradient


    def fit(self,X,y):
        print ("started SGD")
        Lambda1=self.Lambda1
        number_of_examples,number_of_features = len(X),len(X[0])
        if self.Lambda1 is None:
            Lambda1 = 1.0/math.sqrt(number_of_examples)
        self.w = np.zeros(number_of_features)#weights initialization
        iterations = params_L1.iter_factor * number_of_examples
        for t in range(iterations):#itarating over examples
            if t%1000000==0:
                print ("in iteration",t,"out of",iterations)
                sys.stdout.flush()
            lr = 1.0/(t+1)
            random_index = r.randint(0,number_of_examples-1)
            y_k = X[random_index]*y[random_index]
            if not self.check_prediction(y_k):
                self.w = self.w-lr*Lambda1*self.L1_norm_subgradient(number_of_features) + lr*y_k*number_of_examples*self.C - lr*self.Lambda2*self.w
            else:
                self.w = self.w-lr*Lambda1*self.L1_norm_subgradient(number_of_features)- lr*self.Lambda2*self.w
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
        return eval.create_trec_eval_file_opt(test_indices,queries,results,str(self.Lambda1)+"_"+str(self.Lambda2)+"_"+str(self.C),score,self.Lambda1,self.Lambda2,self.C,validation)
