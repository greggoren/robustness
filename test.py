import itertools

a = range(40)
for comb in list(itertools.combinations(a, 35))[30]:
    print(comb)
