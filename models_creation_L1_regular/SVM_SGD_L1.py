import SVM_SGD as svm_s
import numpy as np
import random as r
import math
from models_creation_L1_regular import params_L1
import sys
class svm_sgd_L1(svm_s.svm_sgd):
    def __init__(self, Lambda=None,C=None):
        self.Lambda = Lambda
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

    def get_random_initialization(self,number_of_features):
        vec=[]
        for i in range(number_of_features):
            random = r.random()
            if random < 0.8:
                vec.append(1000)
            else:
                vec.append(5500)
        return vec

    def fit(self,X,y):
        print ("started SGD")

        Lambda=self.Lambda
        number_of_examples,number_of_features = len(X),len(X[0])
        tmp= list(range(number_of_examples))
        # r.shuffle(tmp)
        validation = tmp[:100000]

        if self.Lambda is None:
            Lambda = 1.0/math.sqrt(number_of_examples)
        self.w = np.zeros(number_of_features)
        iterations = params_L1.iter_factor * number_of_examples
        for t in range(iterations):#itarating over examples
            if t%1000000==0:
                print ("in iteration",t,"out of",iterations)
                sys.stdout.flush()
            if t%50000==0:
                error_rate = self.check_validation(validation, y, X)
                print("error_rate is ", error_rate)
                sys.stdout.flush()
            lr = 1.0/(self.Lambda*(t+1))
            random_index = r.randint(0,number_of_examples-1)
            y_k = X[random_index]*y[random_index]
            if not self.check_prediction(y_k):
                self.w = self.w-lr*Lambda*self.L1_norm_subgradient(number_of_features) + lr*y_k*number_of_examples*self.C
            else:
                self.w = self.w-lr*Lambda*self.L1_norm_subgradient(number_of_features)
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
        return eval.create_trec_eval_file_opt(test_indices,queries,results,str(self.Lambda)+"_"+str(self.C),score,self.Lambda,self.C,validation)
