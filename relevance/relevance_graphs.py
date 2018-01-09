import matplotlib.pyplot as plt
import pickle


def create_bar_plot(title, file_name, xlabel, ylabel, stats):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.bar(list(stats.keys()), list(stats.values()), color='b', align='center')
    # plt.hist(stats)
    plt.savefig(file_name)
    plt.clf()


first_two_relevant, histogram_from_rel_to_not, histogram_from_not_to_rel, histogram_from_rel_to_rel, histogram_from_not_to_not = pickle.load(
    open("results_relevance", 'rb'))
create_bar_plot("Percentage of relevant documents top 2 rankings", "first_two", "Epochs", "%", first_two_relevant)
for i in range(2, 9):
    create_bar_plot("Relevant to non-relevant", "rel_not" + str(i), "Rank", "#", histogram_from_rel_to_not[i])
    create_bar_plot("Non-relevant to non-relevant", "not_not" + str(i), "Rank", "#", histogram_from_not_to_not[i])
    create_bar_plot("Relevant to relevant", "rel_rel" + str(i), "Rank", "#", histogram_from_rel_to_rel[i])
    create_bar_plot("Non-relevant to relevant", "not_rel" + str(i), "Rank", "#", histogram_from_not_to_rel[i])
