import SVM_SGD as svm_s
import numpy as np
import random as r
import math
import evaluator_ent
import params_abs as params_ent
import sys
import itertools
class svm_sgd_abs(svm_s.svm_sgd):
    def __init__(self, C=None,Gamma =None):
        self.C = C
        self.w = None
        self.Gamma = Gamma

    def absolute_value(self, number_of_features):
        addition = np.zeros(number_of_features)
        for i,j in itertools.combinations(range(number_of_features),2):
            if self.w[i] - self.w[j] >= 0:
                addition[i]+=1
                addition[j]-=1
            else:
                addition[i] -= 1
                addition[j] += 1
        return addition


    def safe_ln(self,x):
        if x <= 0:
            return 0
        return math.log(x)


    def fit(self,X,y):
        r.seed(params_ent.random_seed)#traceability reasons
        print ("started SGD")
        number_of_examples,number_of_features = len(X),len(X[0])
        self.w = np.zeros(number_of_features)#weights initialization
        if self.C is not None:
            lambda_factor = self.C*number_of_examples
        else:
            lambda_factor = number_of_examples
            self.C=0
        iterations = params_ent.iter_factor * number_of_examples
        for t in range(iterations):#itarating over examples
            if t%1000000==0:
                print ("in iteration",t,"out of",iterations)
                sys.stdout.flush()
            lr = 1.0/(t+1)

            random_index = r.randint(0,number_of_examples-1)
            y_k = X[random_index]*y[random_index]
            absolute_value_part =self.absolute_value(number_of_features)
            if not self.check_prediction(y_k):
                self.w = t*lr*self.w + lr*lambda_factor*y_k-self.Gamma*absolute_value_part*lr
            else:
                self.w = t * lr * self.w - self.Gamma*absolute_value_part*lr

        print ("SGD ended")


    def predict(self,X,queries,test_indices,eval,validation=None):
        results = {}
        for index in test_indices:
            results[index] = np.dot(self.w,X[index].T)
        return eval.create_trec_eval_file(test_indices,queries,results,str(self.C)+"_"+str(self.Gamma),validation)

    def predict_opt(self,X,queries,test_indices,eval,score,gamma,validation=None):
        results = {}
        for index in test_indices:
            results[index] = np.dot(self.w,X[index].T)
        return eval.create_trec_eval_file_opt(test_indices,queries,results,str(self.C)+"_"+str(self.Gamma),score,gamma,validation)
