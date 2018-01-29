import pickle
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau
from scipy.stats.mstats import spearmanr
from numpy import corrcoef
from scipy.stats import linregress
import pandas
def create_scatter_plot(title, file_name, xlabel, ylabel, x, y):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter(x, y, color='g')
    plt.savefig(file_name)
    plt.clf()


# C,mean,max = pickle.load(open("mean_max_c_w_kt",'rb'))
# trees,mean,max = pickle.load(open("mean_max_trees_w_kt",'rb'))
C, kt = pickle.load(open("kt_norm_last", 'rb'))
C, wc = pickle.load(open("wc_norm_last", 'rb'))

# create_scatter_plot("Weighted kendall tau (max aggregation) as function of C","max_36_45","C","Weighted KT",C[36:45],max[36:45])
# create_scatter_plot("Weighted kendall tau (mean aggregation) as function of C","mean_36_45","C","Weighted KT",C[36:46],mean[36:46])
# create_scatter_plot("Weighted kendall tau (mean aggregation) as function of #trees","mean_tress","#trees","Weighted KT",trees,mean)
# create_scatter_plot("Weighted kendall tau (max aggregation) as function of #trees","max_tress","#trees","Weighted KT",trees,max)
create_scatter_plot("WC as a function of ||w|| on  competition data", "wc_scater_on_norm", "||w||",
                    "WC", C, wc)
create_scatter_plot("KT as function of ||w|| on competition data", "kt_scater_on_norm", "||w||",
                    "KT", C, kt)


print(len(C))
print(len(wc))
print(pearsonr(C, wc))
print(spearmanr(C, wc, use_ties=False))
print(kendalltau(wc, C))
b = [C, wc]
df = pandas.DataFrame.from_records(b)
print(df.corr("pearson"))
# create_scatter_plot("Weighted kendall tau (mean aggregation) as function of #leaves", "mean_leaves", "#leaves",
#                     "Weighted KT", leaves, mean)
