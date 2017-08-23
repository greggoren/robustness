import SVM_SGD as svm_s
import numpy as np
import random as r
class svm_sgd_entropy(svm_s.svm_sgd):
    def __init__(self, C=None,Gamma =None):
        self.C = C
        self.w = None
        self.Gamma = Gamma

    def entropy_part_for_sgd(self,number_of_features):
        ""

    def fit(self,X,y):
        print ("started SGD")
        number_of_examples,number_of_features = len(X),len(X[0])
        self.w = np.zeros(number_of_features)#weights initialization
        if self.C is not None:
            lambda_factor = self.C*number_of_examples
        else:
            lambda_factor = number_of_examples
        iterations = 2 * number_of_examples
        for t in range(iterations):#itarating over examples
            if t%1000000==0:
                print ("in iteration",t,"out of",iterations)
            lr = 1.0/(t+1)

            random_index = r.randint(0,number_of_examples-1)
            y_k = X[random_index]*y[random_index]
            if not self.check_prediction(y_k):
                self.w = t*lr*self.w + lr*lambda_factor*y_k
            else:
                self.w = t * lr * self.w

        print ("SGD ended")
