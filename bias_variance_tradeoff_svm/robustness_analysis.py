import pickle
import numpy as np

lb_stats = pickle.load(open("lb_robustness_stats", 'rb'))
svm_stats = pickle.load(open("svm_robustness_stats", 'rb'))
percentages = {m: {} for m in svm_stats}
total = {m: {"lb": 0, "svm": 0} for m in svm_stats}
for metric in svm_stats:
    for epoch in svm_stats[metric]:
        if epoch == 1:
            continue
        percentages[metric][epoch] = {"lb": 0, "svm": 0, "d": 0}
        for query in svm_stats[metric][epoch]:
            if svm_stats[metric][epoch][query] > lb_stats[metric][epoch][query]:
                percentages[metric][epoch]["svm"] += 1
            elif svm_stats[metric][epoch][query] < lb_stats[metric][epoch][query]:
                percentages[metric][epoch]["lb"] += 1
            percentages[metric][epoch]["d"] += 1
        percentages[metric][epoch]["svm"] = float(percentages[metric][epoch]["svm"]) / percentages[metric][epoch]["d"]
        percentages[metric][epoch]["lb"] = float(percentages[metric][epoch]["lb"]) / percentages[metric][epoch]["d"]
    total[metric]["lb"] = np.mean([percentages[metric][e]["lb"] for e in percentages[metric]])
    total[metric]["svm"] = np.mean([percentages[metric][e]["svm"] for e in percentages[metric]])

with open("summary.csv", 'w') as s:
    s.write("METRIC,LambdaMART,SVMRank\n")
    for metric in total:
        s.write(metric.upper() + "," + str(total[metric]["lb"]) + "," + str(total[metric]["svm"]) + "\n")
