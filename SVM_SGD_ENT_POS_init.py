import SVM_SGD_ENT_POS as svm_s
import numpy as np
import random as r
import math
import params_ent
import sys
class svm_sgd_entropy_pos_init(svm_s.svm_sgd_entropy_pos):
    def fit(self,X,y,w):
        r.seed(params_ent.random_seed)#traceability reasons
        print ("started SGD")
        number_of_examples,number_of_features = len(X),len(X[0])
        self.w = w#weights initialization
        if self.C is not None:
            lambda_factor = self.C*number_of_examples
        else:
            lambda_factor = number_of_examples
            self.C = 0
        iterations = math.floor(params_ent.iter_factor * number_of_examples)
        for t in range(iterations):#itarating over examples
            if t%1000000==0:
                print ("in iteration",t,"out of",iterations)
                sys.stdout.flush()
            lr = 1.0/(t+1)

            random_index = r.randint(0,number_of_examples-1)
            y_k = X[random_index]*y[random_index]
            if not self.check_prediction(y_k):
                self.w = (1-self.C-self.Gamma)*t*lr*self.w + lr*lambda_factor*y_k-self.Gamma*self.entropy_part_for_sgd(number_of_features)*lr
            else:
                self.w = (1-self.C-self.Gamma)*t * lr * self.w - self.Gamma*self.entropy_part_for_sgd(number_of_features)*lr

        print ("SGD ended")
