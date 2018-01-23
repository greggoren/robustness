import itertools
import random
a = range(40)
for comb in itertools.combinations(a, 35):
    if random.random() < 0.65:
        continue
    print(comb)
