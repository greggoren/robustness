import random
import numpy as np
from sklearn.linear_model.tests.test_passive_aggressive import random_state


def init():
    models = np.random.uniform(0, 10000, 60)
    return list(set(models))


random.seed(9001)
print(init())
