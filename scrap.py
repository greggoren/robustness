import math
import optimizer
import models_handler
import pickle
if __name__=="__main__":
    with open("model_handler.pickle",'rb') as f:
        a = pickle.load(f)
        print(a.weights_index)