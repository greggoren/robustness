
import matplotlib.pyplot as plt
features=["Avg KT" , "Max KT", " Avg RBO" , "Max RBO" ,"WC" , "Min WC" , "Avg NDCG@5","MAP","MRR"]
# features_to_plot=["Avg KT" , "WC" ,"Avg NDCG@5","MAP","MRR"]
features_to_plot = ["WC", "Avg KT", "Avg NDCG@5", "MAP", "MRR"]
not_to_plot = set(features) - set(features_to_plot)
colors = {"Avg KT":"g" , "WC":"k" ,"Avg NDCG@5":"b","MAP":"r","MRR":"m"}
# epsilons = [1,1.5,2,2.5,3,3.5,4,4.5,5]
epsilons = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200]
feature_map = {f: i for i, f in enumerate(features)}
def retrieve_data(file):
    data_epsilon={}
    with open(file) as data:
        for line in data:
            splited = line.split(" & ")
            if splited[0]=="Ranker":
                continue
            for i,value in enumerate(list(splited[2:])):
                if not data_epsilon.get(splited[0],False):
                    data_epsilon[splited[0]]={}
                if not data_epsilon[splited[0]].get(float(splited[1]),False):
                    data_epsilon[splited[0]][float(splited[1])]={}
                data_epsilon[splited[0]][float(splited[1])][i]=float(value.replace(" \\\\","").rstrip())
    return data_epsilon


def create_plot(features, data_epsilon, epsilons, feature_map, colors, file, not_to_plot):
    plt.figure(1)
    plt.title("Statistics")
    plt.xlabel("Epsilon")
    for feature in features:
        if feature not in not_to_plot:
            y = [data_epsilon["LambdaMart"][i][feature_map[feature]] for i in epsilons]
            plt.plot(epsilons, y, label=feature, color=colors[feature])
    plt.legend(loc='best')
    plt.savefig(file)
    plt.clf()


def create_rel_plot(metric, data, epsilons, colors, name):
    plt.figure(1)
    plt.title(metric)
    plt.xlabel("Iterations")

    for epsilon in epsilons:
        y = data[epsilon][metric]
        plt.plot(range(1, 9), y, label=str(epsilon), color=colors[epsilon])
    plt.legend(loc='best')
    plt.savefig(metric + name[0] + "@" + name[1])
    plt.clf()


relevant_epsilons = [0, 20, 40, 50, 100]
colors = {0: "g", 20: "k", 40: "b", 70: "r", 100: "m", 50: "c"}


def read_file(rel, epsilons):
    data = {e: {"NDCG": [], "MAP": [], "MRR": []} for e in epsilons}
    with open(rel) as relevance_file:
        for row in relevance_file:
            splitted = row.split(" & ")
            epsilon = int(splitted[0].split("_")[1])
            metric = splitted[1]
            if metric == "ND":
                metric = "NDCG"
            data[epsilon][metric] = [float(a) for a in splitted[2:]]
    return data








def create_histogram(data_epsilon,title,feature,feature_map,epsilons):
    x_svm=[a+1 for a in epsilons]
    x_lambda_mart= [a-1 for a in epsilons]
    y_svm=[data_epsilon["SVMRank"][i][feature_map[feature]] for i in epsilons]
    y_lambda_mart=[data_epsilon["LambdaMart"][i][feature_map[feature]] for i in epsilons]
    plt.figure(1)
    plt.title(title)
    plt.xlabel("Epsilon")
    plt.ylabel(feature)
    plt.bar(x_svm,y_svm, align="center",width=2,label="SVMRank",color="b")
    plt.bar(x_lambda_mart,y_lambda_mart,width=2, align="center",label="LambdaMart",color="g")
    plt.legend(loc='best',bbox_to_anchor=(0.5, 0.5))
    plt.savefig("../plt/"+feature)
    plt.clf()


# data_epsilon = retrieve_data("table_value_epsilons_LmbdaMart06.tex")
# create_plot(features, data_epsilon, epsilons, feature_map, colors, "epsilon06", not_to_plot)
# for feature in features:
#     create_histogram(data_epsilon,feature,feature,feature_map,epsilons)
agreement = [4, 5]
level = [1, 2, 3]
for a in agreement:
    for l in level:
        data = read_file("relevance_stats_full" + str(a) + "_" + str(l) + ".tex", epsilons)
        for metric in ["NDCG", "MAP", "MRR"]:
            create_rel_plot(metric, data, relevant_epsilons, colors, [str(a), str(l)])
