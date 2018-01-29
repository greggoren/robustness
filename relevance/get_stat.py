import numpy as np
stat ={i:{} for i in range(9)}
res={i:0 for i in range(9)}
counter = 0
with open("documents_updated.rel") as rel:
    for line in rel:
        splited=line.split()
        query = splited[0]
        iter = int(splited[2].split("-")[1])
        if not stat[iter].get(query,False):
            stat[iter][query]=[]
        if int(splited[3])>0:
            stat[iter][query].append(1)
            counter += 1
        else:
            stat[iter][query].append(0)
print(counter)
a={i:{} for i in range(9)}
for i in stat:
    tmp=0
    d = 0
    for q in stat[i]:
        a[i][q]=sum(stat[i][q])
        tmp+=np.mean(stat[i][q])
        d+=1
    res[i]=float(tmp)/d

for i in stat:
    print("iter", i)
    print(sum([1 for q in a[i] if a[i][q] == 1 or a[i][q] == 0]))

