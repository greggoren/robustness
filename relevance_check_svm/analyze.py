import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np


def create_scatter_plot(title, file_name, xlabel, ylabel, x, y):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter(x, y, color='g')
    plt.savefig(file_name)
    plt.clf()


C_array = [(i + 1) / 1000 for i in range(5)]
C_array.extend([(i + 1) / 100 for i in range(5)])
C_array.extend([(i + 1) / 10000 for i in range(5)])
C_array.extend([(i + 1) / 10 for i in range(5)])
C_array.extend([(i + 1) / 1 for i in range(5)])
C_array.extend([(i + 1) * 10 for i in range(5)])
C_array.extend([(i + 1) * 100 for i in range(5)])
C = {}
with open("C_relevance_for_correlation_svm") as C_stats:
    for stat in C_stats:
        if stat.__contains__("MODEL"):
            continue

        model = stat.split()[0]
        if float(model) not in C_array:
            continue
        if not C.get(float(model), False):
            C[float(model)] = {}
        C[float(model)][stat.split()[1]] = float(stat.split()[2].split('\'')[1].rstrip())


models = set(C.keys())
print(set(C_array) - models)

C_keys = sorted(list(C.keys()))
C_map = []
C_ndcg = []
C_p5 = []
C_p10 = []
for key in C_keys:
    C_map.append(C[key]["map"])
    C_ndcg.append(C[key]["ndcg_cut.20"])
    C_p5.append(C[key]["P.5"])
    C_p10.append(C[key]["P.10"])

create_scatter_plot("Map as a function of C", "C_map", "C", "Map", C_keys[10:], C_map[10:])
create_scatter_plot("NDCG@20 as a function of C", "C_ndcg", "C", "NDCG@20", C_keys[10:], C_ndcg[10:])
create_scatter_plot("P@5 as a function of C", "C_p5", "C", "P@5", C_keys[10:], C_p5[10:])
create_scatter_plot("P@10 as a function of C", "C_p10", "C", "P@10", C_keys[10:], C_p10[10:])

f = open("pearson_C_rel.tex", 'w')
f.write("\\begin{tabular}{c|c|c}\n")
f.write("Metric & Correlation & P-value \\\\ \n")
corr = pearsonr(np.array(C_keys[10:]), np.array(C_map[10:]))
f.write("Map & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = pearsonr(C_keys[10:], C_ndcg[10:])
f.write("NDCG@20 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = pearsonr(C_keys[10:], C_p5[10:])
f.write("P@5 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = pearsonr(C_keys[10:], C_p10[10:])
f.write("P@10 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
f.write("\\end{tabular}")
f.close()

f = open("spearman_C_rel.tex", 'w')
f.write("\\begin{tabular}{c|c|c}\n")
f.write("Metric & Correlation & P-value \\\\ \n")
corr = spearmanr(np.array(C_keys[10:]), np.array(C_map[10:]))
f.write("Map & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = spearmanr(C_keys[10:], C_ndcg[10:])
f.write("NDCG@20 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = spearmanr(C_keys[10:], C_p5[10:])
f.write("P@5 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = spearmanr(C_keys[10:], C_p10[10:])
f.write("P@10 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
f.write("\\end{tabular}")
f.close()
