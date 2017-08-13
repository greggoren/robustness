import numpy as np
import random as r
import evaluator as e
class svm_sgd:

    def __init__(self,C=None):
        self.C=C
        self.w = None


    def check_prediction(self,y_k):

        if np.dot(self.w,y_k.T)>1:
            return True
        return False


    def fit(self,X,y):
        print ("started SGD")
        number_of_examples,number_of_features = len(X),len(X[0])
        self.w = np.zeros(number_of_features)#weights initialization
        if self.C is not None:
            lambda_factor = self.C*number_of_examples
        else:
            lambda_factor = number_of_examples

        for t in range(number_of_examples):#itarating over examples
            if t%1000000==0:
                print ("in iteration",t,"out of",number_of_examples)
            lr = 1.0/(t+1)

            random_index = r.randint(0,number_of_examples-1)
            y_k = X[random_index]*y[random_index]
            if not self.check_prediction(y_k):
                self.w = t*lr*self.w + lr*lambda_factor*y_k
            else:
                self.w = t * lr * self.w

        print ("SGD ended")


    def predict(self,X,queries,test_indices,validation=None):

        results = {}

        for index in test_indices:
            results[index] = np.dot(self.w,X[index].T)

        eval = e.eval()
        return eval.create_trec_eval_file(test_indices,queries,results,str(self.C),validation)









