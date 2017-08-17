from scipy.optimize import minimize
from scipy.spatial.distance import cosine
import numpy as np

class optimizer:
    def __init__(self,alpha,w):
        self.alpha = alpha
        self.w = w

    def constraint_one(self,x_0,x):

        return self.alpha-1+cosine(x_0,x)

    def constraint_two(self,x):
        return 1-x

    def constraint_three(self,x):#probably wont be in use
        return x

    def objective(self,x):
        return -np.dot(self.w,x.T)

    def get_best_features(self,x_0):
        cons = [{'type': 'ineq','args':(x_0,) ,'fun':self.constraint_one},{'type': 'ineq', 'fun': lambda x:1-x},{'type': 'ineq', 'fun': lambda x:x}]
        return minimize(self.objective,x_0,method='SLSQP',constraints=cons)