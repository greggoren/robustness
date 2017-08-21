from scipy.optimize import minimize
from scipy.spatial.distance import cosine
import numpy as np

class optimizer:
    def __init__(self,alpha,w):
        self.alpha = alpha
        self.w = w

    def constraint1(self, x_0, x):
        return self.alpha-1+cosine(x_0,x)


    def objective(self,x):
        return -np.dot(self.w,x.T)

    def get_best_features(self,x_0):
        cons = [{'type': 'ineq','args':(x_0,) ,'fun':self.constraint1}, {'type': 'ineq', 'fun': lambda x: 1 - x}, {'type': 'ineq', 'fun': lambda x:x}]
        return minimize(self.objective,x_0,method='SLSQP',constraints=cons)['x']