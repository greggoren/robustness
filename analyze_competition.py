import svm_ent_models_handler
import svm_models_handler
import numpy as np
from scipy.stats import kendalltau
from scipy import spatial
import itertools
import subprocess
import matplotlib.pyplot as plt
import RBO as r
import pickle
def create_plot(title,file_name,xlabel,ylabel,models,index,x_axis):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for svm in models:
        ax.plot(x_axis,models[svm][index],svm[3],label=svm[2])
    #ax.plot( x_axis,svm_ent, 'b', label='svm_ent')
    plt.legend(loc='best')
    plt.savefig(file_name)
    plt.clf()

def create_single_plot(title,file_name,xlabel,ylabel,y,x):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(x,y,'g')
    plt.savefig(file_name)
    plt.clf()


def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    return iter(p.stdout.readline,'')

class analysis:

    def __init__(self):
        ""

    def cosine_distance(self,x,y):
        return spatial.distance.cosine(x,y)

    def get_all_scores(self,svms,competition_data):
        scores = {}
        for svm in svms:
            scores[svm] = {}
            epochs = range(1,9)
            for epoch in epochs:
                scores[svm][epoch] = {}
                for query in competition_data[epoch]:
                    scores[svm][epoch][query] = {}
                    fold = svm[0].query_to_fold_index[int(query)]
                    weights_svm = svm[0].weights_index[fold]
                    for doc in competition_data[epoch][query]:
                        features_vector = competition_data[epoch][query][doc]
                        scores[svm][epoch][query][doc] = np.dot(features_vector, weights_svm.T)
        return scores

    def get_competitors(self,scores_svm):
        competitors={}
        for query in scores_svm[1]:
            competitors[query] = scores_svm[1][query].keys()
        return competitors


    def retrieve_ranking(self,scores):
        rankings_svm = {}

        optimized = False
        for svm in scores:
            if not optimized:
                competitors = self.get_competitors(scores[svm])
                optimized = True
            rankings_svm[svm]={}
            scores_svm = scores[svm]
            for epoch in scores_svm:
                rankings_svm[svm][epoch]={}
                for query in scores_svm[epoch]:
                    retrieved_list_svm = sorted(competitors[query],key=lambda x:scores_svm[epoch][query][x],reverse=True)
                    rankings_svm[svm][epoch][query]= self.transition_to_rank_vector(competitors[query],retrieved_list_svm)
        return rankings_svm

    def transition_to_rank_vector(self,original_list,sorted_list):
        rank_vector = []
        for doc in original_list:
            try:
                rank_vector.append(sorted_list.index(doc) + 1)
            except:
                print(original_list,sorted_list)
        return rank_vector


    def calculate_average_kendall_tau(self, rankings):
        kendall = {}
        change_rate = {}
        rbo_min_models = {}
        for svm in rankings:
            rankings_list_svm = rankings[svm]
            kt_svm = []
            kt_svm_orig = []
            last_list_index_svm={}
            original_list_index_svm = {}
            change_rate_svm_epochs =[]
            rbo_min = []
            rbo_min_orig = []

            for epoch in rankings_list_svm:
                sum_svm = 0
                sum_rbo_min = 0
                sum_rbo_min_orig = 0
                sum_svm_original = 0
                n_q=0
                change_rate_svm = 0
                for query in rankings_list_svm[epoch]:
                    current_list_svm = rankings_list_svm[epoch][query]
                    if not last_list_index_svm.get(query,False):
                        last_list_index_svm[query]=current_list_svm
                        original_list_index_svm[query]=current_list_svm
                        continue
                    if current_list_svm.index(1)!=last_list_index_svm[query].index(1):
                        change_rate_svm +=1
                    n_q+=1
                    kt = kendalltau(last_list_index_svm[query], current_list_svm)[0]
                    kt_orig = kendalltau(original_list_index_svm[query], current_list_svm)[0]
                    d=0
                    rbo_orig_average=0
                    rbo_average = 0
                    for i in range(1):
                        rbo_orig_average+= r.rbo_dict({x:i for x,i in enumerate(original_list_index_svm[query])},{x:i for x,i in enumerate(current_list_svm)} , 0.1)["min"]
                        rbo_average += r.rbo_dict({x:i for x,i in enumerate(last_list_index_svm[query])},{x:i for x,i in enumerate(current_list_svm)},0.1)["min"]
                        d+=1

                    rbo_measure_orig = float(rbo_orig_average)/d
                    rbo_measure = float(rbo_average)/d
                    sum_rbo_min+=rbo_measure
                    sum_rbo_min_orig+=rbo_measure_orig
                    if not np.isnan(kt):
                        sum_svm+=kt
                    if not np.isnan(kt_orig):
                        sum_svm_original+=kt_orig
                    last_list_index_svm[query] = current_list_svm

                if n_q==0:
                    continue
                change_rate_svm_epochs.append(change_rate_svm)
                kt_svm.append(float(sum_svm)/n_q)
                kt_svm_orig.append(float(sum_svm_original)/n_q)
                rbo_min.append(float(sum_rbo_min)/n_q)
                rbo_min_orig.append(float(sum_rbo_min_orig)/n_q)
            kendall[svm]=(kt_svm,kt_svm_orig)
            rbo_min_models[svm] = (rbo_min,rbo_min_orig)
            change_rate[svm]=(change_rate_svm_epochs,)
        return kendall,change_rate,rbo_min_models,range(2,9)

    def calcualte_average_distances(self,competition_data):
        average_distances =[]
        for epoch in competition_data:
            total_average_distance_sum =0
            number_of_queries = 0
            for query in competition_data[epoch]:
                number_of_queries+=1
                sum_distance_query = 0
                denom = 0
                docs = competition_data[epoch][query].keys()
                for doc1,doc2 in itertools.combinations(docs,2):
                    sum_distance_query+=self.cosine_distance(competition_data[epoch][query][doc1],competition_data[epoch][query][doc2])
                    denom+=1
                total_average_distance_sum+=float(sum_distance_query)/denom
            average_distances.append(total_average_distance_sum/number_of_queries)
        return average_distances

    def set_qid_for_trec(self,query):
        if query < 10:
            qid = "00" + str(query)
        elif query < 100:
            qid = "0" + str(query)
        else:
            qid = str(query)
        return qid



    def extract_score(self, scores):
        for svm in scores:
            for epoch in scores[svm]:
                part = svm[1].split(".pickle")
                name = part[0]+part[1].replace(".","")+svm[2]
                f = open(name+str(epoch)+".txt",'w')
                for query in scores[svm][epoch]:
                    for doc in scores[svm][epoch][query]:
                        f.write(self.set_qid_for_trec(int(query))+" Q0 "+"ROUND-0"+str(epoch)+"-"+self.set_qid_for_trec(int(query))+"-"+doc+" "+str(0) +" "+ str(scores[svm][epoch][query][doc])+" seo\n")
                f.close()

    def calculate_metrics(self,models):
        metrics = {}
        for svm in models:
            ndcg_by_epochs = []
            map_by_epochs = []
            for i in range(1,9):
                part = svm[1].split(".pickle")
                name = part[0] + part[1].replace(".", "")+svm[2]
                score_file =  name+str(i)+".txt"
                qrels = "rel/rel0"+str(i)+".txt"
                command = "./trec_eval -m ndcg_cut.5 "+qrels+" "+score_file
                for line in run_command(command):
                    ndcg_score = line.split()[2].rstrip()
                    ndcg_by_epochs.append(ndcg_score)
                    break
                command1 = "./trec_eval -m map " + qrels + " " + score_file
                for line in run_command(command1):
                    print(line)
                    map_score = line.split()[2].rstrip()
                    map_by_epochs.append(map_score)
                    break
            metrics[svm] = (ndcg_by_epochs,map_by_epochs)
        return metrics

    def get_average_epsilon(self,scores,number_of_competitors):
        stat={}
        for svm in scores:
            deltas = []
            for epoch in scores[svm]:
                sum = 0
                denom = 0
                for query in scores[svm][epoch]:
                    scores_list = sorted([scores[svm][epoch][query][doc] for doc in scores[svm][epoch][query]])
                    last_score = None

                    for s in scores_list:
                        if last_score is None:
                            last_score = s
                            continue
                        else:
                            sum += (s-last_score)
                            last_score=s
                            denom+=1
                deltas.append(float(sum)/denom)
            stat[svm]=(deltas,)
        return stat

    def fix_ranking(self,svm,query,scores,epsilon,epoch,current_ranking,last_ranking):
        new_rank = []
        for rank in range(len(current_ranking)):
            if rank+1<len(current_ranking):
                doc_win =current_ranking[rank]
                doc_lose = current_ranking[rank+1]
                if doc_win=='32' or doc_lose=='32':
                    print("here")
                if last_ranking.index(doc_lose)<last_ranking.index(doc_win) and (scores[svm][epoch][query][doc_win]-scores[svm][epoch][query][doc_lose])<epsilon and  (doc_lose not in new_rank):
                    new_rank.append(doc_lose)
                    if doc_win not in new_rank:
                        new_rank.append(doc_win)
                elif doc_win not in new_rank:
                    new_rank.append(doc_win)

            else:
                if current_ranking[rank] not in new_rank:
                    new_rank.append(current_ranking[rank])
                elif current_ranking[rank-1] not in new_rank:
                    new_rank.append(current_ranking[rank-1])
        return new_rank

    def rerank_by_epsilon(self,svm,scores,epsilon):
        rankings_svm = {}
        new_scores ={}
        last_rank = {}
        competitors = self.get_competitors(scores[svm])
        rankings_svm[svm] = {}
        scores_svm = scores[svm]
        for epoch in scores_svm:
            rankings_svm[svm][epoch] = {}
            new_scores[epoch] = {}
            for query in scores_svm[epoch]:
                retrieved_list_svm = sorted(competitors[query], key=lambda x: scores_svm[epoch][query][x],
                                            reverse=True)

                if not last_rank.get(query,False):
                    last_rank[query] = retrieved_list_svm
                fixed = self.fix_ranking(svm,query,scores,epsilon,epoch,retrieved_list_svm,last_rank[query])

                rankings_svm[svm][epoch][query] = self.transition_to_rank_vector(competitors[query],fixed)
                last_rank[query] = fixed
                new_scores[epoch][query] = {x:(5-fixed.index(x)) for x in fixed}
        scores[svm] = new_scores
        return rankings_svm[svm],scores



    def analyze(self, svms, competition_data, dump):
        scores = self.get_all_scores(svms,competition_data)
        rankings_svm = self.retrieve_ranking(scores)
        k = open("epslon_kt.txt",'a')
        r =open("epsilon_rbo.txt",'a')
        c=open("epsilon_winner_change.txt",'a')
        #positive_negative
        k.write("Model,Min Kendall-Tau,Max Kendall-Tau,Average Kendall-Tau\n")
        c.write("Model,Min Winner Change,Max Winner Change,Average Winner Change\n")
        r.write("Model,Min RBO,Max RBO,Average RBO\n")
        for svm in svms:
            if svm[2]=="svm_epsilon":
                rankings_svm[svm],scores = self.rerank_by_epsilon(svm,scores,1.5)
            if not dump:

                kendall, cr, rbo_min, x_axis = self.calculate_average_kendall_tau(rankings_svm)
                k.write(svm[2]+","+str(min(kendall[svm][0]))+","+str(max(kendall[svm][0]))+","+str(float(sum(kendall[svm][0]))/len((kendall[svm][0])))+"\n")
                c.write(svm[2]+","+str(min(cr[svm][0]))+","+str(max(cr[svm][0]))+","+str(float(sum(cr[svm][0]))/len((cr[svm][0])))+"\n")
                r.write(svm[2]+","+str(min(rbo_min[svm][0]))+","+str(max(rbo_min[svm][0]))+","+str(float(sum(rbo_min[svm][0]))/len((rbo_min[svm][0])))+"\n")
                """create_plot("Average Kendall-Tau with last iteration","plt/kt1_reg.PNG","Epochs","Kendall-Tau",kendall,0,x_axis)
                create_plot("Average Kendall-Tau with original list","plt/kt1_orig_reg.PNG","Epochs","Kendall-Tau",kendall,1,x_axis)
                create_plot("Average RBO measure with original list","plt/rbo1_min_orig_reg.PNG","Epochs","RBO",rbo_min,1,x_axis)
                create_plot("Average RBO measure with last iteration","plt/rbo1_min_reg.PNG","Epochs","RBO",rbo_min,0,x_axis)
                create_plot("Number of queries with winner changed", "plt/winner_change1_reg.PNG", "Epochs", "#Queries",cr,0, x_axis)"""
                # deltas = self.get_average_epsilon(number_of_competitors=5,scores=scores)
                # create_plot("Average epsilon by epoch", "plt/eps.PNG", "Epochs", "Average epsilon", deltas, 0,  range(1,9))
                # with open("comp_reg.pickle", 'rb') as f:
                #     metrics = pickle.load(f)
                #     create_plot("NDCG@5 by epochs", "plt/ndcg_reg.png", "Epochs", "NDCG@5", metrics, 0, range(1, 9))
                #     create_plot("map@5 by epochs", "plt/map_reg.png", "Epochs", "map@5", metrics, 1, range(1, 9))
            else:
                self.extract_score(scores)
                metrics=self.calculate_metrics(scores)
                with open("comp.pickle",'wb') as f:
                    pickle.dump(metrics,f)
        k.close()
        c.close()
        r.close()