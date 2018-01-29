import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
import pickle

def create_scatter_plot(title, file_name, xlabel, ylabel, x, y):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter(x, y, color='g')
    plt.savefig(file_name)
    plt.clf()


def recover_model(model):
    indexes_covered = []
    weights = []
    with open(model) as model_file:
        for line in model_file:
            if line.__contains__(":"):
                wheights = line.split()
                wheights_length = len(wheights)

                for index in range(1, wheights_length - 1):

                    feature_id = int(wheights[index].split(":")[0])
                    if index < feature_id:
                        for repair in range(index, feature_id):
                            if repair in indexes_covered:
                                continue
                            weights.append(0)
                            indexes_covered.append(repair)
                    weights.append(float(wheights[index].split(":")[1]))
                    indexes_covered.append(feature_id)
    return np.array(weights)


def map_between_C_and_norm(C_array):
    mapping = {}
    for C in C_array:
        sum_norm = 0
        for fold in range(1, 6):
            model_file = "models/" + str(fold) + "/svm_model" + str(C) + ".txt"
            w = recover_model(model_file)
            sum_norm += np.linalg.norm(w)
        mapping[C] = sum_norm / 5
    return mapping

# C_array = [(i + 1) / 1000 for i in range(5)]
# C_array.extend([(i + 1) / 100 for i in range(5)])
# C_array.extend([(i + 1) / 10000 for i in range(5)])
# C_array.extend([(i + 1) / 10 for i in range(5)])
# C_array.extend([(i + 1) / 1 for i in range(5)])
# C_array.extend([(i + 1) * 10 for i in range(5)])
# C_array.extend([(i + 1) * 100 for i in range(5)])
C = {}
with open("C_relevance_for_correlation_svm") as C_stats:
    for stat in C_stats:
        if stat.__contains__("MODEL"):
            continue

        model = stat.split()[0]

        if not C.get(float(model), False):
            C[float(model)] = {}
        C[float(model)][stat.split()[1]] = float(stat.split()[2].split('\'')[1].rstrip())


models = set(C.keys())
# print(set(C_array) - models)
mapping = map_between_C_and_norm(models)
print(mapping)
C_keys1 = sorted(list(C.keys()))
C_keys = []
C_map = []
C_ndcg = []
C_p5 = []
C_p10 = []
for key in C_keys1:
    C_keys.append(mapping[key])
    C_map.append(C[key]["map"])
    C_ndcg.append(C[key]["ndcg_cut.20"])
    C_p5.append(C[key]["P.5"])
    C_p10.append(C[key]["P.10"])

# create_scatter_plot("Map as a function of C", "C_map", "C", "Map", C_keys, C_map)
# create_scatter_plot("NDCG@20 as a function of C", "C_ndcg", "C", "NDCG@20", C_keys, C_ndcg)
# create_scatter_plot("P@5 as a function of C", "C_p5", "C", "P@5", C_keys, C_p5)
# create_scatter_plot("P@10 as a function of C", "C_p10", "C", "P@10", C_keys, C_p10)

f = open("pearson_C_rel.tex", 'w')
f.write("\\begin{tabular}{c|c|c}\n")
f.write("Metric & Correlation & P-value \\\\ \n")
corr = pearsonr(np.array(C_keys), np.array(C_map))
d = open("a", 'wb')
pickle.dump((C_keys, C_map))
d.close()
f.write("Map & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = pearsonr(C_keys, C_ndcg)
f.write("NDCG@20 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = pearsonr(C_keys, C_p5)
f.write("P@5 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = pearsonr(C_keys, C_p10)
f.write("P@10 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
f.write("\\end{tabular}")
f.close()

f = open("spearman_C_rel.tex", 'w')
f.write("\\begin{tabular}{c|c|c}\n")
f.write("Metric & Correlation & P-value \\\\ \n")
corr = spearmanr(np.array(C_keys), np.array(C_map))
f.write("Map & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = spearmanr(C_keys, C_ndcg)
f.write("NDCG@20 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = spearmanr(C_keys, C_p5)
f.write("P@5 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = spearmanr(C_keys, C_p10)
f.write("P@10 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
f.write("\\end{tabular}")
f.close()
