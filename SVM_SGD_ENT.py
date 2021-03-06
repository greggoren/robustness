import SVM_SGD as svm_s
import numpy as np
import random as r
import math
import evaluator_ent
import params_ent
import sys
class svm_sgd_entropy(svm_s.svm_sgd):
    def __init__(self, C=None,Gamma =None):
        self.C = C
        self.w = None
        self.Gamma = Gamma

    def entropy_part_for_sgd(self,number_of_features):
        r_t, z_t = 0, 0
        for i in self.w:
            r_t += (i ** 2) * self.safe_ln(i**2)
            z_t += i ** 2
        addition = np.zeros(number_of_features)
        if z_t!=0:#avoid division by zero
            for i,w_i in enumerate(self.w):
                addition[i] = w_i*((self.safe_ln(w_i**2))/z_t - r_t/(z_t**2))
        return addition


    def safe_ln(self,x):
        if x <= 0:
            return 0
        return math.log(x)


    def fit(self,X,y):
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
                ent = self.entropy_part_for_sgd(number_of_features)
                print(ent)
                sys.stdout.flush()
            lr = 1.0/(t+1)

            random_index = r.randint(0,number_of_examples-1)
            y_k = X[random_index]*y[random_index]
            ent =self.entropy_part_for_sgd(number_of_features)
            if not self.check_prediction(y_k):
                # self.w = (t+self.C+self.Gamma)*lr*self.w + lr*lambda_factor*y_k-self.Gamma*ent*lr
                self.w = t*lr*self.w + lr*lambda_factor*y_k-self.Gamma*ent*lr
            else:
                # self.w = (t+self.C+self.Gamma)* lr * self.w - self.Gamma*ent*lr
                self.w = t * lr * self.w - self.Gamma*ent*lr
        print ("SGD ended")


    def predict(self,X,queries,test_indices,eval,validation=None):
        results = {}
        for index in test_indices:
            results[index] = np.dot(self.w,X[index].T)
        return eval.create_trec_eval_file(test_indices,queries,results,str(self.C)+"_"+str(self.Gamma),validation)

    def predict_opt(self,X,queries,test_indices,eval,score,validation=None):
        results = {}
        for index in test_indices:
            results[index] = np.dot(self.w,X[index].T)
        return eval.create_trec_eval_file_opt(test_indices,queries,results,str(self.C)+"_"+str(self.Gamma),score,self.Gamma,validation)
