import math
import optimizer
if __name__=="__main__":
    opt= optimizer.optimizer(0.001, [1,2,3])
    x_0=[0.1,0,0.3]
    res = opt.get_best_features(x_0)
    print(res['x'])