import itertools
import random

a = range(10)
for comb in itertools.combinations(a, 3):
    # if random.random() < 0.8:
    #     continue
    print(comb)
