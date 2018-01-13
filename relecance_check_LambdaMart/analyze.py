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


trees = {}
with open("trees_relevance_for_correlation_LambdaMart") as trees_stats:
    for stat in trees_stats:
        if stat.__contains__("MODEL"):
            continue

        model = stat.split("\t")[0]
        if not trees.get(int(model.split("_")[0]), False):
            trees[int(model.split("_")[0])] = {}
        trees[int(model.split("_")[0])][stat.split("\t")[1]] = float(stat.split("\t")[2].split('\'')[1].rstrip())

tree_keys = sorted(list(trees.keys()))
trees_map = []
trees_ndcg = []
trees_p5 = []
trees_p10 = []
for key in tree_keys:
    trees_map.append(trees[key]["map"])
    trees_ndcg.append(trees[key]["ndcg_cut.20"])
    trees_p5.append(trees[key]["P.5"])
    trees_p10.append(trees[key]["P.10"])

create_scatter_plot("Map as a function of #trees", "trees_map", "#trees", "Map", tree_keys, trees_map)
create_scatter_plot("NDCG@20 as a function of #trees", "trees_ndcg", "#trees", "NDCG@20", tree_keys, trees_ndcg)
create_scatter_plot("P@5 as a function of #trees", "trees_p5", "#trees", "P@5", tree_keys, trees_p5)
create_scatter_plot("P@10 as a function of #trees", "trees_p10", "#trees", "P@10", tree_keys, trees_p10)

f = open("pearson_trees_rel.tex", 'w')
f.write("\\begin{tabular}{c|c|c}\n")
f.write("Metric & Correlation & P-value \\\\ \n")
corr = pearsonr(np.array(tree_keys), np.array(trees_map))
f.write("Map & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = pearsonr(tree_keys, trees_ndcg)
f.write("NDCG@20 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = pearsonr(tree_keys, trees_p5)
f.write("P@5 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = pearsonr(tree_keys, trees_p10)
f.write("P@10 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
f.write("\\end{tabular}")
f.close()

f = open("spearman_trees_rel.tex", 'w')
f.write("\\begin{tabular}{c|c|c}\n")
f.write("Metric & Correlation & P-value \\\\ \n")
corr = spearmanr(np.array(tree_keys), np.array(trees_map))
f.write("Map & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = spearmanr(tree_keys, trees_ndcg)
f.write("NDCG@20 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = spearmanr(tree_keys, trees_p5)
f.write("P@5 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = spearmanr(tree_keys, trees_p10)
f.write("P@10 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
f.write("\\end{tabular}")
f.close()

leaves = {}
with open("leaves_relevance_for_correlation_LambdaMart") as leaves_stats:
    for stat in leaves_stats:
        if stat.__contains__("MODEL"):
            continue

        model = stat.split("\t")[0]
        if not leaves.get(int(model.split("_")[1]), False):
            leaves[int(model.split("_")[1])] = {}
        leaves[int(model.split("_")[1])][stat.split("\t")[1]] = float(stat.split("\t")[2].split('\'')[1].rstrip())

leaves_keys = sorted(list(leaves.keys()))
leaves_map = []
leaves_ndcg = []
leaves_p5 = []
leaves_p10 = []
for key in leaves_keys:
    leaves_map.append(leaves[key]["map"])
    leaves_ndcg.append(leaves[key]["ndcg_cut.20"])
    leaves_p5.append(leaves[key]["P.5"])
    leaves_p10.append(leaves[key]["P.10"])

create_scatter_plot("Map as a function of #leaves", "leaves_map", "#leaves", "Map", leaves_keys, leaves_map)
create_scatter_plot("NDCG@20 as a function of #leaves", "leaves_ndcg", "#leaves", "NDCG@20", leaves_keys, leaves_ndcg)
create_scatter_plot("P@5 as a function of #leaves", "leaves_p5", "#leaves", "P@5", leaves_keys, leaves_p5)
create_scatter_plot("P@10 as a function of #leaves", "leaves_p10", "#leaves", "P@10", leaves_keys, leaves_p10)

f = open("pearson_leaves_rel.tex", 'w')
f.write("\\begin{tabular}{c|c|c}\n")
f.write("Metric & Correlation & P-value \\\\ \n")
corr = pearsonr(np.array(leaves_keys), np.array(leaves_map))
f.write("Map & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = pearsonr(leaves_keys, leaves_ndcg)
f.write("NDCG@20 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = pearsonr(leaves_keys, leaves_p5)
f.write("P@5 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = pearsonr(leaves_keys, leaves_p10)
f.write("P@10 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
f.write("\\end{tabular}")
f.close()

f = open("spearman_leaves_rel.tex", 'w')
f.write("\\begin{tabular}{c|c|c}\n")
f.write("Metric & Correlation & P-value \\\\ \n")
corr = spearmanr(np.array(leaves_keys), np.array(leaves_map))
f.write("Map & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = spearmanr(leaves_keys, leaves_ndcg)
f.write("NDCG@20 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = spearmanr(leaves_keys, leaves_p5)
f.write("P@5 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
corr = spearmanr(leaves_keys, leaves_p10)
f.write("P@10 & " + str(round(corr[0], 3)) + " & " + str(round(corr[1], 3)) + " \\\\ \n")
f.write("\\end{tabular}")
f.close()

"""C={}
with open("C...") as C_stats:
    for stat in C_stats:
        if stat.__contains__("MODEL"):
            continue

        model = stat.split("\t")[0]
        if not C.get(float(model),False):
            C[float(model)]={}
        C[float(model)][stat.split("\t")[1]]=float(stat.split("\t")[2].split('\'')[1].rstrip())


C_keys = sorted(list(C.keys()))
C_map=[]
C_ndcg=[]
C_p5=[]
C_p10=[]
for key in C_keys:
    C_map.append(C[key]["map"])
    C_ndcg.append(C[key]["ndcg_cut.20"])
    C_p5.append(C[key]["P.5"])
    C_p10.append(C[key]["P.10"])

create_scatter_plot("Map as a function of #C","C_map","#C","Map",C_keys,C_map)
create_scatter_plot("NDCG@20 as a function of #C","C_ndcg","#C","NDCG@20",C_keys,C_ndcg)
create_scatter_plot("P@5 as a function of #C","C_p5","#C","P@5",C_keys,C_p5)
create_scatter_plot("P@10 as a function of #C","C_p10","#C","P@10",C_keys,C_p10)

f = open("pearson_C_rel.tex",'w')
f.write("\\begin{tabular}{c|c|c}\n")
f.write("Metric & Correlation & P-value \\\\ \n")
corr= pearsonr(np.array(C_keys),np.array(C_map))
f.write("Map & "+str(round(corr[0],3))+" & "+ str(round(corr[1],3))+" \\\\ \n")
corr=pearsonr(C_keys,C_ndcg)
f.write("NDCG@20 & "+str(round(corr[0],3))+" & "+ str(round(corr[1],3))+" \\\\ \n")
corr= pearsonr(C_keys,C_p5)
f.write("P@5 & "+str(round(corr[0],3))+" & "+ str(round(corr[1],3))+" \\\\ \n")
corr= pearsonr(C_keys,C_p10)
f.write("P@10 & "+str(round(corr[0],3))+" & "+ str(round(corr[1],3))+" \\\\ \n")
f.write("\\end{tabular}")
f.close()

f = open("spearman_C_rel.tex",'w')
f.write("\\begin{tabular}{c|c|c}\n")
f.write("Metric & Correlation & P-value \\\\ \n")
corr= spearmanr(np.array(C_keys),np.array(C_map))
f.write("Map & "+str(round(corr[0],3))+" & "+ str(round(corr[1],3))+" \\\\ \n")
corr=spearmanr(C_keys,C_ndcg)
f.write("NDCG@20 & "+str(round(corr[0],3))+" & "+ str(round(corr[1],3))+" \\\\ \n")
corr= spearmanr(C_keys,C_p5)
f.write("P@5 & "+str(round(corr[0],3))+" & "+ str(round(corr[1],3))+" \\\\ \n")
corr= spearmanr(C_keys,C_p10)
f.write("P@10 & "+str(round(corr[0],3))+" & "+ str(round(corr[1],3))+" \\\\ \n")
f.write("\\end{tabular}")
f.close()

"""
