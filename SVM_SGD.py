import numpy as np
import random as r
import params
import evaluator
import sys
class svm_sgd:

    def __init__(self,C=None):
        self.C=C
        self.w = None


    def check_prediction(self,y_k):

        if np.dot(self.w,y_k.T)>1:
            return True
        return False


    def check_validation(self,validation,tags,X):
        errors=0

        for index in validation:
            y_k=X[index]*tags[index]
            tmp=np.dot(self.w,y_k.T)
            if tmp<1:
                errors+=1

        return float(errors)/len(validation)

    def fit(self,X,y):
        print("started SGD")
        number_of_examples,number_of_features = len(X),len(X[0])
        tmp = list(range(number_of_examples))
        # r.shuffle(tmp)
        validation = tmp[:100000]
        self.w = np.zeros(number_of_features)#weights initialization
        if self.C is not None:
            lambda_factor = self.C*number_of_examples
        else:
            lambda_factor = number_of_examples
        iterations = params.iter_factor * number_of_examples
        for t in range(iterations):#itarating over examples
            if t%1000000==0:
                print ("in iteration",t,"out of",iterations)
            lr = 1.0/(t+1)
            if t%50000==0:
                error_rate = self.check_validation(validation, y, X)
                print("error_rate is ", error_rate)
                sys.stdout.flush()
            random_index = r.randint(0,number_of_examples-1)
            y_k = X[random_index]*y[random_index]
            if not self.check_prediction(y_k):
                self.w = t*lr*self.w + lr*lambda_factor*y_k
            else:
                self.w = t * lr * self.w

        print ("SGD ended")



    def predict(self,X,queries,test_indices,eval,validation=None):
        results = {}
        for index in test_indices:
            results[index] = np.dot(self.w,X[index].T)
        return eval.create_trec_eval_file(test_indices,queries,results,str(self.C),validation)









