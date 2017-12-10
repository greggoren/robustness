
import matplotlib.pyplot as plt
features=["Avg KT" , "Max KT", " Avg RBO" , "Max RBO" ,"WC" , "Min WC" , "Avg NDCG@5"]
# epsilons = [1,1.5,2,2.5,3,3.5,4,4.5,5]
epsilons = [0, 10, 20, 30, 40, 50, 60, 70,80,90,100]
feature_map = {f:i for i,f in enumerate(features)}
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

data_epsilon = retrieve_data("table_value_epsilons_LmbdaMart_per.tex")
for feature in features:
    create_histogram(data_epsilon,feature,feature,feature_map,epsilons)