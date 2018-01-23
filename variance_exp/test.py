import pickle

a = pickle.load(open("variance_data", 'rb'))
for s in a:
    for i in a[s]:
        d = a[s][i]
        d = [float(k.split()[-1].rstrip()) for k in d]
        a[s][i] = d
with open("variance_data1", 'wb') as data:
    pickle.dump(a, data)
