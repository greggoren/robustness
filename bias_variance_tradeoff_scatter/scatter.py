import pickle
import matplotlib.pyplot as plt


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
C, kt = pickle.load(open("kt", 'rb'))
C, wc = pickle.load(open("wc", 'rb'))

# create_scatter_plot("Weighted kendall tau (max aggregation) as function of C","max_36_45","C","Weighted KT",C[36:45],max[36:45])
# create_scatter_plot("Weighted kendall tau (mean aggregation) as function of C","mean_36_45","C","Weighted KT",C[36:46],mean[36:46])
# create_scatter_plot("Weighted kendall tau (mean aggregation) as function of #trees","mean_tress","#trees","Weighted KT",trees,mean)
# create_scatter_plot("Weighted kendall tau (max aggregation) as function of #trees","max_tress","#trees","Weighted KT",trees,max)
create_scatter_plot("WC as f(C) Competition - Last ", "wc_scat_2", "C",
                    "WC", C, wc)
create_scatter_plot("KT as f(C) Competition - Last", "kt_scat_2", "C",
                    "KT", C, kt)

# create_scatter_plot("Weighted kendall tau (mean aggregation) as function of #leaves", "mean_leaves", "#leaves",
#                     "Weighted KT", leaves, mean)
