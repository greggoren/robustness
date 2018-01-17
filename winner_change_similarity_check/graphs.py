import matplotlib.pyplot as plt
import pickle


def create_bar_plot(title, file_name, xlabel, ylabel, stats):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    print(sorted(list(stats.keys()), key=lambda x: x[0]))
    indexes = [list(stats.keys()).index(i) + 1 for i in sorted(list(stats.keys()), key=lambda x: x[0])]

    plt.bar(indexes, list(stats.values()), color='b', align='center')
    # plt.hist(stats)
    plt.savefig(file_name)
    plt.clf()


bins_for_new_winner_self_similarity, bins_for_winner_similarity, total_self, total_to_winner = pickle.load(
    open("bins_stats_lm", 'rb'))
for epoch in bins_for_new_winner_self_similarity:
    create_bar_plot("Similarity to previous vector on winner change event in epoch" + str(epoch),
                    "self_sim" + str(epoch), "Bin", "%", bins_for_new_winner_self_similarity[epoch])
    create_bar_plot("Similarity to former winner on winner change event in epoch" + str(epoch),
                    "fwinner_sim" + str(epoch), "Bin", "%", bins_for_winner_similarity[epoch])
create_bar_plot("Similarity to previous vector on winner change event total", "total_self_sim_lm", "Bin", "%",
                total_self)
create_bar_plot("Similarity to former winner vector on winner change event total", "total_winner_sim_lm", "Bin", "%",
                total_to_winner)
