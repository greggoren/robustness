from copy import copy
import svm_ent_models_handler
import numpy as np
import svm_models_handler
import numpy as np
from scipy.stats import kendalltau
from scipy import spatial
import itertools
import subprocess
import matplotlib.pyplot as plt
import RBO as r
import RBO_EXT as rob
import pickle
import math
import os
from sklearn.datasets import dump_svmlight_file

def write_files(svms,kendall,cr,rbo_min):
    k = open("results/epsilon1_kt.txt",'a')
    r =open("results/epsilon1_rbo.txt",'a')
    c=open("results/epsilon1_wc.txt",'a')
    k.write("Model,Min Kendall-Tau,Max Kendall-Tau,Average Kendall-Tau\n")
    c.write("Model,Min Winner Change,Max Winner Change,Aversage Winner Change\n")
    r.write("Model,Min RBO,Max RBO,Average RBO\n")
    for svm in svms:
        k.write(svm[2].replace("_"," ")+","+str(round(min(kendall[svm][0]),3))+","+str(round(max(kendall[svm][0]),3))+","+str(round(float(sum(kendall[svm][0]))/len((kendall[svm][0])),3))+"\n")
        c.write(svm[2].replace("_"," ")+","+str(round(min(cr[svm][0]),3))+","+str(round(max(cr[svm][0]),3))+","+str(round(float(sum(cr[svm][0]))/len((cr[svm][0])),3))+"\n")
        r.write(svm[2].replace("_"," ")+","+str(round(min(rbo_min[svm][0]),3))+","+str(round(max(rbo_min[svm][0]),3))+","+str(round(float(sum(rbo_min[svm][0]))/len((rbo_min[svm][0])),3))+"\n")
    k.close()
    c.close()
    r.close()



def create_plot(title,file_name,xlabel,ylabel,models,index,x_axis):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for svm in models:
        ax.plot(x_axis,models[svm][index],svm[3],label=svm[2])
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


def run_bash_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,shell=True)
    out,err = p.communicate()
    print(out)
    return out
    #return iter(p.stdout.readline,'')
def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,shell=True
                         )
    return iter(p.stdout.readline,'')


def cosine_similarity(v1, v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i];
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


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
                        scores[svm][epoch][query][doc] = np.dot(weights_svm,features_vector.T )
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
                    retrieved_list_svm = sorted(competitors[query],key=lambda x:(scores_svm[epoch][query][x],x),reverse=True)
                    rankings_svm[svm][epoch][query]= self.transition_to_rank_vector(competitors[query],retrieved_list_svm)
        return rankings_svm

    def transition_to_rank_vector(self,original_list,sorted_list):
        rank_vector = []
        for doc in original_list:
            try:
                rank_vector.append(6-(sorted_list.index(doc)+1))
                # rank_vector.append(sorted_list.index(doc) + 1)
            except:
                print(original_list,sorted_list)
        return rank_vector


    def get_average_weights_similarity_from_baseline(self,baseline,model,folds):

        sum_of_similarity = 0
        for fold in folds:
            sum_of_similarity+=cosine_similarity(baseline[0][0].weights_index[fold],model.weights_index[fold])
        return str(round(sum_of_similarity/len(folds),3))

    def rbo_with_all_p(self,list1,list2,values = [0.25,0.5,0.75,0.9]):
        result = {}
        for p in values:
            result[p] = r.rbo_dict({x:j for x,j in enumerate(list1)},{x:j for x,j in enumerate(list2)},p)["min"]
        return result

    def calculate_average_kendall_tau(self, rankings,values):
        kendall = {}
        change_rate = {}
        rbo_min_models = {}
        meta_rbo = {}
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
                sum_rbo_ps = {p:0 for p in values}
                meta_rbo[svm]={p:[] for p in values}
                for query in rankings_list_svm[epoch]:
                    current_list_svm = rankings_list_svm[epoch][query]
                    if not last_list_index_svm.get(query,False):
                        last_list_index_svm[query]=current_list_svm
                        original_list_index_svm[query]=current_list_svm
                        continue
                    if current_list_svm.index(5)!=last_list_index_svm[query].index(5):
                        change_rate_svm +=1
                    n_q+=1
                    kt = kendalltau(last_list_index_svm[query], current_list_svm)[0]
                    kt_orig = kendalltau(original_list_index_svm[query], current_list_svm)[0]
                    rbo_orig= r.rbo_dict({x:j for x,j in enumerate(original_list_index_svm[query])},{x:j for x,j in enumerate(current_list_svm)} , 0.95)["min"]
                    rbo = r.rbo_dict({x:j for x,j in enumerate(last_list_index_svm[query])},{x:j for x,j in enumerate(current_list_svm)},0.95)["min"]
                    tmp = self.rbo_with_all_p(last_list_index_svm[query],current_list_svm,values)
                    sum_rbo_ps={p:sum_rbo_ps[p]+tmp[p] for p in values}
                    sum_rbo_min+=rbo
                    sum_rbo_min_orig+=rbo_orig
                    if not np.isnan(kt):
                        sum_svm+=kt#*weights[epoch][query]
                    if not np.isnan(kt_orig):
                        sum_svm_original+=kt_orig
                    last_list_index_svm[query] = current_list_svm

                if n_q==0:
                    continue
                change_rate_svm_epochs.append(float(change_rate_svm)/n_q)
                for p in values:
                    meta_rbo[svm][p].append(sum_rbo_ps[p]/n_q)
                kt_svm.append(float(sum_svm)/n_q)
                kt_svm_orig.append(float(sum_svm_original)/n_q)
                rbo_min.append(float(sum_rbo_min)/n_q)
                rbo_min_orig.append(float(sum_rbo_min_orig)/n_q)
            kendall[svm]=(kt_svm,kt_svm_orig)
            rbo_min_models[svm] = (rbo_min,rbo_min_orig)
            change_rate[svm]=(change_rate_svm_epochs,)
        return kendall,change_rate,rbo_min_models,range(2,9),meta_rbo


    def create_rbo_table(self,svms,kendall,meta_rbo,values,names,table_file):
        for svm in svms:
            results = [str(round(sum(meta_rbo[svm][p])/len(meta_rbo[svm][p]),4)) for p in values]
            model = svm[2].split("_")
            gamma = model[0]
            if len(model) == 1:
                sigma = "-"
            else:
                sigma = model[1]
            k = str(round(sum(kendall[svm][0])/len(kendall[svm][0]),4))
            line =names[svm[1].split("/")[0]] + " & " + gamma + " & " + sigma + " & "+k+" & " +" & ".join(results)+" \\\\ \n"
            table_file.write(line)

    def create_rbo_table_baseline(self,svms,kendall,meta_rbo,values,table_file):
        for svm in svms:
            results = [str(round(sum(meta_rbo[svm][p])/len(meta_rbo[svm][p]),4)) for p in values]
            k = str(round(sum(kendall[svm][0])/len(kendall[svm][0]),4))
            line ="" + " & " + "-" + " & " + "-" + " & "+k+" & " +" & ".join(results)+" \\\\ \n"
            table_file.write(line)

    def create_rbo_table_max_baseline(self,svms,kendall,meta_rbo,values,table_file):
        for svm in svms:
            results = [str(round(max(meta_rbo[svm][p]),4)) for p in values]
            k = str(round(max(kendall[svm][0]),4))
            line ="" + " & " + "-" + " & " + "-" + " & "+k+" & " +" & ".join(results)+" \\\\ \n"
            table_file.write(line)

    def create_rbo_table_max(self,svms,kendall,meta_rbo,values,names,table_file):
        for svm in svms:
            results = [str(round(max(meta_rbo[svm][p]),4)) for p in values]
            model = svm[2].split("_")
            gamma = model[0]
            if len(model) == 1:
                sigma = "-"
            else:
                sigma = model[1]
            k = str(round(max(kendall[svm][0]),4))
            line =names[svm[1].split("/")[0]] + " & " + gamma + " & " + sigma + " & "+k+" & " +" & ".join(results)+" \\\\ \n"
            table_file.write(line)

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


    def calculate_average_distance_from_last_iteration(self,competition_data):
        weights_for_average = {}
        average_similarity = []
        original_similarity = []
        for epoch in competition_data:
            total_epoch_sum = 0
            nq = 0
            if epoch == 1:
                continue
            sum_average_of_query = 0
            sum_average_original_of_query = 0
            weights_for_average[epoch]={}
            for query in competition_data[epoch]:
                nq += 1
                sum_similarity_query = 0
                sum_original_similarity_query = 0

                for doc in competition_data[epoch][query]:

                    sum_similarity_query +=(cosine_similarity(competition_data[epoch][query][doc],
                                                              competition_data[epoch - 1][query][doc]))
                    sum_original_similarity_query +=(cosine_similarity(competition_data[epoch][query][doc],
                                                              competition_data[1][query][doc]))
                total_epoch_sum+=sum_similarity_query
                sum_average_of_query += (float(sum_similarity_query)/5)
                sum_average_original_of_query += sum_original_similarity_query/5
                weights_for_average[epoch][query]= sum_similarity_query
            for query in weights_for_average[epoch]:
                weights_for_average[epoch][query]=weights_for_average[epoch][query]/total_epoch_sum
            average_similarity.append(float(sum_average_of_query)/nq)
            original_similarity.append(sum_average_original_of_query/nq)
        return average_similarity,original_similarity,weights_for_average


    def get_average_weight_differences(self,w,indexes):
        denominator = 0
        sum_of_differences = 0
        for i,j in itertools.combinations(indexes, 2):
            sum_of_differences+=abs(w[i]-w[j])
            denominator+=1
        return float(sum_of_differences)/denominator

    def analyze_weights(self,svms):
        for svm in svms:
            mh = svm[0]
            print(svm[2])
            for i in range(1,6):
                print("fold ",i)
                print(self.get_average_weight_differences(mh.weights_index[i],range(25)))


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

    def determine_order(self,pair,current_ranking):
        if current_ranking.index(pair[0])<current_ranking.index(pair[1]):
            return pair[0],pair[1]
        else:
            return pair[1],pair[0]

    def fix_ranking(self,svm,query,scores,epsilon,epoch,current_ranking,last_ranking,model):
        new_rank =[]
        if model==1:
            for rank in range(len(current_ranking)):
                if rank + 1 < len(current_ranking):
                    if not new_rank:
                        doc_win =current_ranking[rank]
                    else:
                        doc_win=new_rank[rank]

                    doc_lose = current_ranking[rank+1]
                    if doc_win in new_rank:
                        new_rank = new_rank[:-1]
                if last_ranking.index(doc_lose) < last_ranking.index(doc_win) and (
                    scores[svm][epoch][query][doc_win] - scores[svm][epoch][query][doc_lose]) < epsilon:
                    new_rank.append(doc_lose)
                    new_rank.append(doc_win)
                else:
                    new_rank.append(doc_win)
                    new_rank.append(doc_lose)

        if model==0:
            for rank in range(len(current_ranking)):
                if rank+1<len(current_ranking):
                    doc_win =current_ranking[rank]
                    doc_lose = current_ranking[rank+1]
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
        if model==2:
            condorcet_count = {doc: 0 for doc in current_ranking}
            doc_pairs = list(itertools.combinations(current_ranking,2))
            for pair in doc_pairs:
                doc_win,doc_lose=self.determine_order(pair,current_ranking)
                if last_ranking.index(doc_lose) < last_ranking.index(doc_win) and (
                    scores[svm][epoch][query][doc_win] - scores[svm][epoch][query][doc_lose]) < epsilon:
                    condorcet_count[doc_lose]+=1
                else:
                    condorcet_count[doc_win]+=1
            new_rank = sorted(current_ranking,key=lambda x:(condorcet_count[x],len(last_ranking)-last_ranking.index(x)),reverse = True)

        return new_rank

    def rerank_by_epsilon(self,svm,scores,epsilon,model):
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
                retrieved_list_svm = sorted(competitors[query], key=lambda x: (scores_svm[epoch][query][x],x),
                                            reverse=True)

                if not last_rank.get(query,False):
                    last_rank[query] = retrieved_list_svm
                fixed = self.fix_ranking(svm,query,scores,epsilon,epoch,retrieved_list_svm,last_rank[query],model)

                rankings_svm[svm][epoch][query] = self.transition_to_rank_vector(competitors[query],fixed)
                last_rank[query] = fixed
                new_scores[epoch][query] = {x:(5-fixed.index(x)) for x in fixed}
        scores[svm] = new_scores
        return rankings_svm[svm],scores



    def add_baselines(self,file,baselines_model_object,kendall,rbo_min,cr,score_dict,names):
        for svm in baselines_model_object:
            model = svm[2].split("_")
            gamma = model[0]
            if len(model) == 1:
                sigma = "-"
            else:
                sigma = model[1]

            file.write(names[svm[1].split("/")[0]] + " & " + gamma + " & " + sigma + " & $" + str(
                round(float(sum(kendall[svm][0])) / len((kendall[svm][0])), 3)) + \
                             "$ & $" + str(round(max(kendall[svm][0]), 3)) + "$ & $" + str(
                round(float(sum(rbo_min[svm][0])) / len((rbo_min[svm][0])), 3)) + "$ & $" + \
                             str(round(max(rbo_min[svm][0]), 3)) + "$ & $" + str(
                round(float(sum(cr[svm][0])) / len(cr[svm][0]), 3)) + "$ & $" + str(
                round(min(cr[svm][0]), 3)) + "$ & $" + score_dict[svm[1].split("/")[0]][svm[2]] + "$ & - \\\\  \n")
            file.write("\hline\n")


    def create_table(self, meta_svms, competition_data,names,score_dict=None,baseline=None,baselines_model_object=None):
        values=[0.25,0.5,0.75,0.9,0.95]
        table_file = open("out/table_big_extra_doubly_c.tex",'w')
        table_best = open("out/bestbig_extra_doubly_c.tex",'w')
        #table_rbo = open("out/rbo_table_big.tex",'w')
        #table_rbo_max = open("out/rbo_table_max_big.tex",'w')
        table_file.write("\\begin{longtable}{*{9}{c}\n")#to \linewidth {l|X[3,c]X[3,c]X[3,c]X[3,c]X[4,c]X[4,c]X[5,c]X[5,c]X[5,c]X[5,r]}\n")
        table_best.write("\\begin{longtable}{ccccc}\n")
        #table_rbo.write("\\begin{longtable}{ccccccccc}\n")
        #table_rbo_max.write("\\begin{longtable}{ccccccccc}\n")
        table_best.write("Ranker & Metric & $\gamma$ & $\sigma$ & Value \\\\\\\\ \n")
        table_file.write(
            "Ranker & $\gamma$ & $\sigma$ & Avg KT & Max KT & Avg RBO & Max RBO & WC & Min WC  \\\\\\\\ \n")
        #table_rbo_max.write("Ranker & $\gamma$ & $\sigma$ Max KT & "+" & ".join([str(p) for p in values])+"\n")
        #table_rbo_max.write("\hline\n")
        #table_rbo.write("Ranker & $\gamma$ & $\sigma$ Avg KT & "+" & ".join([str(p) for p in values])+"\n")
        #table_rbo.write("\hline\n")
        for svms in meta_svms:
            table_best.write("\hline\n")

            # table_file.write(
            #     "Ranker & $\gamma$ & $\sigma$ & Avg KT & Max KT & Avg RBO & Max RBO & WC & NDCG & SIM \\\\\\\\ \n")
            table_file.write("\hline\n")
            scores = self.get_all_scores(svms, competition_data)
            rankings_svm = self.retrieve_ranking(scores)
            for svm in svms:
                if svm[2] == "svm_epsilon":
                    rankings_svm[svm], scores = self.rerank_by_epsilon(svm, scores, 1.5)
            kendall, cr, rbo_min, x_axis,meta_rbo = self.calculate_average_kendall_tau(rankings_svm,values)
            # self.create_rbo_table(svms,kendall,meta_rbo,values,names,table_rbo)
            # self.create_rbo_table_max(svms,kendall,meta_rbo,values,names,table_rbo_max)
            #self.add_baselines(table_file, baselines_model_object, kendall, rbo_min, cr, score_dict, names)

            a_kt = []
            m_kt = []
            a_rbo =[]
            m_rbo =[]
            a_cr =[]
            nd = []
            m_cr = []
            for svm in svms:
                # sim = self.get_average_weights_similarity_from_baseline(baseline,svm[0],range(1,6))
                model=svm[2].split("_")
                gamma = model[0]
                if len(model)==1:
                    sigma = "-"
                else:
                    sigma = model[1]

                kt_avg = float(sum(kendall[svm][0]))/len((kendall[svm][0]))
                a_kt.append((svm,(gamma,sigma),kt_avg))
                max_kt = max(kendall[svm][0])
                m_kt.append((svm,(gamma,sigma),max_kt))
                avg_rbo = float(sum(rbo_min[svm][0]))/len((rbo_min[svm][0]))
                a_rbo.append((svm,(gamma,sigma),avg_rbo))
                max_rbo = max(rbo_min[svm][0])
                m_rbo.append((svm,(gamma,sigma),max_rbo))
                change = float(sum(cr[svm][0]))/len(cr[svm][0])
                m_change = min(cr[svm][0])
                a_cr.append((svm,(gamma,sigma),change))
                m_cr.append((svm,(gamma,sigma),m_change))
                # ndcg = "-"#score_dict[svm[1].split("/")[0]][svm[2]]
                # nd.append((svm,(gamma,sigma),ndcg))
                table_file.write(names[svm[1].split("/")[0]]+ " & "+gamma+" & "+sigma+" & $"+str(round(float(sum(kendall[svm][0]))/len((kendall[svm][0])),3))+\
                                 "$ & $"+str(round(max(kendall[svm][0]),3))+"$ & $"+str(round(float(sum(rbo_min[svm][0]))/len((rbo_min[svm][0])),3))+"$ & $"+\
                                 str(round(max(rbo_min[svm][0]),3))+"$ & $"+str(round(float(sum(cr[svm][0]))/len(cr[svm][0]),3))+"$ & $"+str(round(min(cr[svm][0]),3))+"  \\\\  \n")
            table_file.write("\hline\n")
            svm,model,v = sorted(a_kt,key=lambda x:x[2],reverse=True)[0]
            table_best.write(names[svm[1].split("/")[0]] + " & Average Kendall-$\\tau$ & "+model[0] + " & " + model[1]+" & "+ str(
                round(float(sum(kendall[svm][0])) / len((kendall[svm][0])), 3))+" \\\\ \n")

            svm,model,v = sorted(m_kt,key=lambda x:x[2],reverse=True)[0]
            table_best.write(names[svm[1].split("/")[0]] + " & Max Kendall-$\\tau$ & "+model[0] + " & " + model[1] +" & "+ str(
                round(max(kendall[svm][0]), 3))+" \\\\ \n")
            svm,model,v = sorted(a_rbo,key=lambda x:x[2],reverse=True)[0]
            table_best.write(names[svm[1].split("/")[0]] + " & Average RBO & " +model[0] + " & " + model[1]+" & "+str(
                round(float(sum(rbo_min[svm][0])) / len((rbo_min[svm][0])), 3))+" \\\\ \n")
            svm,model,v = sorted(m_rbo,key=lambda x:x[2],reverse=True)[0]
            table_best.write(names[svm[1].split("/")[0]] + " & Max RBO & " + model[0] + " & " + model[1]+" & "+ str(
                round(max(rbo_min[svm][0]), 3))+" \\\\ \n")
            svm,model,v = sorted(a_cr,key=lambda x:x[2],reverse=False)[0]
            table_best.write(names[svm[1].split("/")[0]] + " & Average Winner Change Ratio & "  + model[0] + " & " + model[1]+" & "+ str(
                round(float(sum(cr[svm][0])) / len((cr[svm][0])), 3))+" \\\\ \n")
            svm, model, v = sorted(m_cr, key=lambda x: x[2], reverse=False)[0]
            table_best.write(
                names[svm[1].split("/")[0]] + " & Min Winner Change Ratio & " + model[0] + " & " + model[1] + " & " + str(round(min(cr[svm][0]),3)) + " \\\\ \n")
            # svm,model,v = sorted(nd,key=lambda x:x[2],reverse=True)[0]
            # table_best.write(names[svm[1].split("/")[0]] + " & Max NDCG@20 & "+ model[0] + " & " + model[1]+" & - \\\\ \n")

        table_file.write("\end{longtable}")
        table_best.write("\end{longtable}")
        # table_rbo.write("\end{longtabu}")
        # table_rbo_max.write("\end{longtabu}")
        table_file.close()
        table_best.close()
        # table_rbo.close()
        # table_rbo_max.close()

    def read_retrieval_scores(self,dir):
        scores_dict = {}
        scores_dict["pos_minus"]={}
        scores_dict["pos_plus"]={}
        scores_dict["squared_minus"]={}
        scores_dict["squared_plus"]={}
        for root,dirs,files in os.walk(dir):
            for file in files:
                file_name = root+"/"+file
                with open(file_name,'r') as scores:
                    for score in scores:
                        if score.split()[0].__contains__("ndcg"):
                            res = score.split()[1].replace("'","").replace("b","")
                            if file.__contains__("pos_sigma_minus"):
                                scores_dict["pos_minus"][file.split(".txt")[1]] = res
                            elif file.__contains__("pos_sigma"):
                                scores_dict["pos_plus"][file.split(".txt")[1]] = res
                            elif file.__contains__("plus"):
                                scores_dict["squared_plus"][file.split(".txt")[1]] = res
                            else:
                                scores_dict["squared_minus"][file.split(".txt")[1]] = res
        return scores_dict


    def analyze(self, svms, competition_data, dump):
        d,o,w = self.calculate_average_distance_from_last_iteration(competition_data)
        # create_single_plot("Average similarity with last iteration", "plt/similarity_last.PNG", "Epochs", "Similarity", d, range(2,9))
        # create_single_plot("Average similarity with original document", "plt/similarity_orig.PNG", "Epochs", "Similarity", o, range(2,9))
        # self.analyze_weights(svms)
        scores = self.get_all_scores(svms,competition_data)
        rankings_svm = self.retrieve_ranking(scores)
        for svm in svms:
            if svm[2].__contains__("svm_epsilon"):
                epsilon = float(svm[2].split("_")[2])
                rankings_svm[svm],scores = self.rerank_by_epsilon(svm,scores,epsilon,0)
        if not dump:
            kendall, cr, rbo_min, x_axis,a = self.calculate_average_kendall_tau(rankings_svm,[])

            # write_files(svms,kendall,cr,rbo_min)
            # create_plot("Average Kendall-Tau with last iteration","plt/kt_epsilon2.PNG","Epochs","Kendall-Tau",kendall,0,x_axis)
            # create_plot("Average Kendall-Tau with original list","plt/kt_orig_epsilon2.PNG","Epochs","Kendall-Tau",kendall,1,x_axis)
            # create_plot("Average RBO measure with original list","plt/rbo_min_orig_eps2.PNG","Epochs","RBO",rbo_min,1,x_axis)
            # create_plot("Average RBO measure with last iteration","plt/rbo_min_epsilon2.PNG","Epochs","RBO",rbo_min,0,x_axis)
            # create_plot("Number of queries with winner changed", "plt/winner_change_epsilon2.PNG", "Epochs", "#Queries",cr,0, x_axis)
            deltas = self.get_average_epsilon(number_of_competitors=5,scores=scores)
            create_plot("Average epsilon by epoch", "plt/eps.PNG", "Epochs", "Average epsilon", deltas, 0,  range(1,9))
            # with open("comp_epsilon1.pickle", 'rb') as f:
            #     metrics = pickle.load(f)
            #     create_plot("NDCG@5 by epochs", "plt/ndcg_epsilon1.png", "Epochs", "NDCG@5", metrics, 0, range(1, 9))
            #     create_plot("map@5 by epochs", "plt/map_epsilon1.png", "Epochs", "map@5", metrics, 1, range(1, 9))
        else:
            self.extract_score(scores)
            metrics=self.calculate_metrics(scores)
            with open("comp_epsilon0.pickle",'wb') as f:
                pickle.dump(metrics,f)
    def create_rbo_baseline(self,svms,competition_data):
        values = [0.25, 0.5, 0.75, 0.9, 0.95]
        table_rbo = open("out/rbo_table_base.tex", 'w')
        table_rbo_max = open("out/rbo_table_max_base.tex", 'w')
        scores = self.get_all_scores(svms, competition_data)
        rankings_svm = self.retrieve_ranking(scores)
        for svm in svms:
            if svm[2] == "svm_epsilon":
                rankings_svm[svm], scores = self.rerank_by_epsilon(svm, scores, 1.5,0)
        kendall, cr, rbo_min, x_axis, meta_rbo = self.calculate_average_kendall_tau(rankings_svm, values)
        self.create_rbo_table_baseline(svms, kendall, meta_rbo, values, table_rbo)
        self.create_rbo_table_max_baseline(svms, kendall, meta_rbo, values, table_rbo_max)
        table_rbo.close()
        table_rbo.close()

    def create_table_for_epsilons(self,epsilons,competition_data):

        names = {0:"Naive",1:"Basic",2:"Modified"}
        metrics = {0:pickle.load(open("comp_epsilon0.pickle", 'rb')),1:pickle.load(open("comp_epsilon1.pickle", 'rb')),2:pickle.load(open("comp_epsilon2.pickle", 'rb'))}
        table_file = open("out/table_value_epsilons.tex", 'w')
        table_file.write("\\begin{longtable}{*{11}{c}}\n")
        table_file.write(
            "Ranker & Avg KT & Max KT & Avg RBO & Max RBO & WC & Min WC & Avg NDCG@5 & Max NDCG@5 & Avg MAP@5 & Max MAP@5 \\\\\\\\ \n")

        for epsilon_model in range(3):
            scores = self.get_all_scores(epsilons, competition_data)
            rankings_svm = self.retrieve_ranking(scores)
            for svm in epsilons:
                if svm[2].__contains__("svm_epsilon"):
                    epsilon = float(svm[2].split("_")[2])
                    rankings_svm[svm], scores = self.rerank_by_epsilon(svm, scores, epsilon,epsilon_model)
            kendall, cr, rbo_min, x_axis, a = self.calculate_average_kendall_tau(rankings_svm, [])

            a_kt = []
            m_kt = []
            a_rbo = []
            m_rbo = []
            a_cr = []
            nd = []
            m_cr = []
            for svm in epsilons:
                sim = "-"

                model = names[epsilon_model]+"_"+svm[2]

                kt_avg = float(sum(kendall[svm][0])) / len((kendall[svm][0]))
                a_kt.append((svm, kt_avg))
                max_kt = max(kendall[svm][0])
                m_kt.append((svm, max_kt))
                avg_rbo = float(sum(rbo_min[svm][0])) / len((rbo_min[svm][0]))
                a_rbo.append((svm, avg_rbo))
                max_rbo = max(rbo_min[svm][0])
                m_rbo.append((svm, max_rbo))
                change = float(sum(cr[svm][0])) / len(cr[svm][0])
                m_change = min(cr[svm][0])
                a_cr.append((svm, change))
                m_cr.append((svm, m_change))
                for k in metrics[epsilon_model]:
                    if k[2]==svm[2]:
                        a_ndcg = round(sum([float(a) for a in metrics[epsilon_model][k][0]])/len(metrics[epsilon_model][k][0]),3)
                        a_map = round(sum([float(a) for a in metrics[epsilon_model][k][1]])/len(metrics[epsilon_model][k][1]),3)
                        m_ndcg = round(max([float(a) for a in metrics[epsilon_model][k][0]]),3)
                        m_map = round(max([float(a) for a in metrics[epsilon_model][k][1]]),3)
                #nd.append((svm, ndcg))

                table_file.write(model.replace("_"," ")+" & $" + str(
                    round(float(sum(kendall[svm][0])) / len((kendall[svm][0])), 3)) + \
                                 "$ & $" + str(round(max(kendall[svm][0]), 3)) + "$ & $" + str(
                    round(float(sum(rbo_min[svm][0])) / len((rbo_min[svm][0])), 3)) + "$ & $" + \
                                 str(round(max(rbo_min[svm][0]), 3)) + "$ & $" + str(
                    round(float(sum(cr[svm][0])) / len(cr[svm][0]), 3)) + "$ & $" + str(
                    round(min(cr[svm][0]), 3)) + "$ & $" + str(a_ndcg) + "$ & $" + str(m_ndcg) +"$ & $"+str(a_map)+"$ & $"+str(m_map)+"$ \\\\  \n")

        table_file.write("\hline\n")
        table_file.write("\\end{longtable}\n")
        table_file.close()

    def get_metrices_for_table(self,meta_svms,competition_data):
        for svms in meta_svms:
            scores = self.get_all_scores(svms, competition_data)
            self.extract_score(scores)
            metrics = self.calculate_metrics(scores)
            name = svms[0][1].split("/")[0]
            with open(name+".pickle", 'wb') as f:
                pickle.dump(metrics, f)

    def append_data_retrieval(self, metrics1,metrics2, names):
        file = open("out/table_minmax.tex",'w')
        with open("out/table_big_extra.tex") as table:
            for line in table:
                if line.__contains__("Min Max"):
                    tmp=copy(line.split(" & ")[:3])
                    tmp[0]=names[tmp[0]]
                    table_model = "_".join(tmp)
                    for metric in metrics2:
                        name = metric[1].split("/")[0]+"_"+"_".join(metric[2].split("_")[1:])
                        if name==table_model:
                            new_line = line.replace("\\\\","")+" & $"+str(round(np.mean([float(a) for a in metrics2[metric][0]]),4))+"$"
                            file.write(new_line+" \\\\ \n ")
                elif line.__contains__("Max Min"):
                    tmp=copy(line.split(" & ")[:3])
                    tmp[0]=names[tmp[0]]
                    table_model = "_".join(tmp)
                    for metric in metrics1:
                        name = metric[1].split("/")[0]+"_"+"_".join(metric[2].split("_")[1:])

                        if name==table_model:
                            new_line = line.replace("\\\\ ","")+" & $"+str(round(np.mean([float(a) for a in metrics1[metric][0]]),4))+"$"
                            file.write(new_line+" \\\\ \n ")
                else:
                    file.write(line)
            file.close()



    def run_lambda_mart(self,features,epoch):
        java_path = "/lv_local/home/sgregory/jdk1.8.0_121/bin/java"
        jar_path = "/lv_local/home/sgregory/SEO_CODE/model_running/RankLib.jar"
        score_file = "/lv_local/home/sgregory/robustness/coodinate_ascent/score"+str(epoch)
        features= "/lv_local/home/sgregory/robustness/"+features
        model_path  = "/lv_local/home/sgregory/robustness/model"
        #os.popen('/bin/bash touch '+score_file)
        command = java_path+" -jar "+jar_path + " -load "+model_path+" -rank "+features+ " -score "+score_file
        run_bash_command(command)
        return score_file

    def create_lambdaMart_scores(self,competition_data):
        scores={e :{} for e in competition_data}
        for epoch in competition_data:
            scores[epoch]={q:{} for q in competition_data[epoch]}
            order = {_e: {} for _e in competition_data}
            data_set = []
            queries=[]
            index = 0
            for query in competition_data[epoch]:
                for doc in competition_data[epoch][query]:
                    data_set.append(competition_data[epoch][query][doc])
                    queries.append(query)
                    order[epoch][index]=doc+"@@@"+query
                    index+=1
            features_file = "features"+str(epoch)
            self.create_data_set_file(data_set,queries,features_file)
            score_file=self.run_lambda_mart(features_file,epoch)
            scores=self.retrieve_scores(score_file,order,epoch,scores)
        return scores

    def create_data_set_file(self,X,queries,feature_file_name):
        with open(feature_file_name,'w') as feature_file:
            for i,doc in enumerate(X):
                features = " ".join([str(a+1)+":"+str(b) for a,b in enumerate(doc)])
                line = "1 qid:"+queries[i]+" "+features+"\n"
                feature_file.write(line)


    def retrieve_scores(self,score_file,order,epoch,result):
        with open(score_file,'r') as scores:
            index = 0
            for score in scores:
                value = float(score.split()[2])
                doc,query = tuple(order[epoch][index].split("@@@"))
                result[epoch][query][doc]=value
                index+=1
        return result

    def get_results(self,scores):
        rankings = self.retrieve_ranking(scores)
        kendall, cr, rbo_min, x_axis, a = self.calculate_average_kendall_tau(rankings, [])

    def create_epsilon_for_Lambda_mart(self, competition_data,svm):
        scores = {}
        tmp = self.create_lambdaMart_scores(competition_data)
        tmp2= self.get_all_scores(svm,competition_data)

        rankings = self.retrieve_ranking(scores)
        epsilons = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        for epsilon in epsilons:
            key_lambdaMart = ("", "l.pickle1", "LambdaMart" + "_" + str(epsilon), "b")
            key_svm = ("", "l.pickle1", "SVMRank" + "_" + str(epsilon), "b")
            scores[key_lambdaMart] = tmp
            scores[key_svm] = tmp2
            rankings[key_lambdaMart], scores = self.rerank_by_epsilon(key_lambdaMart, scores, epsilon, 2)
            rankings[key_svm], scores = self.rerank_by_epsilon(key_svm, scores, epsilon, 2)
        kendall, cr, rbo_min, x_axis, a = self.calculate_average_kendall_tau(rankings, [])
        self.extract_score(scores)
        metrics = self.calculate_metrics(scores)
        table_file = open("out/table_value_epsilons_LmbdaMart.tex", 'w')
        table_file.write("\\begin{longtable}{*{8}{c}}\n")
        table_file.write(
            "Ranker & Avg KT & Max KT & Avg RBO & Max RBO & WC & Min WC & Avg NDCG@5\\\\\\\\ \n")
        for key_lambdaMart in kendall:
            kt_avg = str(round(np.mean(kendall[key_lambdaMart][0]),3))
            max_kt = str(round(max(kendall[key_lambdaMart][0]),3))
            avg_rbo = str(round(np.mean(rbo_min[key_lambdaMart][0]),3))
            max_rbo =str(round(max(rbo_min[key_lambdaMart][0]),3))
            change = str(round(np.mean(cr[key_lambdaMart][0]),3))
            m_change =str(round(min(cr[key_lambdaMart][0]),3))
            nd=str(round(np.mean([float(a) for a in metrics[key_lambdaMart][0]]),3))
            tmp=[kt_avg,max_kt,avg_rbo,max_rbo,change,m_change,nd]
            line=key_lambdaMart[2]+" & "+" & ".join(tmp)+" \\\\ \n"
            table_file.write(line)

        table_file.write("\\end{longtable}")


    def compare_rankers(self,svm,competition_data):
        scores = self.get_all_scores(svm,competition_data)
        scores[("","l.pickle1","LambdaMart","b")] = self.create_lambdaMart_scores(competition_data)
        self.extract_score(scores)
        metrics = self.calculate_metrics(scores)
        for metric in metrics:
            print(metric)
            print(sum([float(a) for a in metrics[metric][0]])/len(metrics[metric][0]))
        rankings = self.retrieve_ranking(scores)
        results = self.calculate_average_kendall_tau(rankings, [])
        with open("results.pickle",'wb') as res:
            pickle.dump(results,res)

    def create_comparison_plots(self,results,svm):
        with open(results,'rb') as res:
            kendall, cr, rbo_min, x_axis, a = pickle.load(res)
            kt_avg = float(sum(kendall[("","l.pickle1","LambdaMart","b")][0])) / len((kendall[("","l.pickle1","LambdaMart","b")][0]))
            max_kt = max(kendall[("","l.pickle1","LambdaMart","b")][0])
            avg_rbo = float(sum(rbo_min[("","l.pickle1","LambdaMart","b")][0])) / len((rbo_min[("","l.pickle1","LambdaMart","b")][0]))
            max_rbo = max(rbo_min[("","l.pickle1","LambdaMart","b")][0])
            change = float(sum(cr[("","l.pickle1","LambdaMart","b")][0])) / len(cr[("","l.pickle1","LambdaMart","b")][0])
            m_change = min(cr[("","l.pickle1","LambdaMart","b")][0])
            print("LAMBDA_MART")
            print ("kt=",kt_avg)
            print ("max kt=",max_kt)
            print ("avg rbo =",avg_rbo)
            print ("max rbo=",max_rbo)
            print ("cr = ",change)
            print ("min cr=",m_change)
            for key in kendall:
                if key!=("","l.pickle1","LambdaMart","b"):
                    kt_avg = float(sum(kendall[key][0])) / len(
                        (kendall[key][0]))
                    max_kt = max(kendall[key][0])
                    avg_rbo = float(sum(rbo_min[key][0])) / len(
                        (rbo_min[key][0]))
                    max_rbo = max(rbo_min[key][0])
                    change = float(sum(cr[key][0])) / len(
                        cr[key][0])
                    m_change = min(cr[key][0])
                    print("SVM")
                    print("kt=", kt_avg)
                    print("max kt=", max_kt)
                    print("avg rbo =", avg_rbo)
                    print("max rbo=", max_rbo)
                    print("cr = ", change)
                    print("min cr=", m_change)
            """
            create_plot("Average Kendall-Tau with last iteration","plt/kt_cmp.PNG","Epochs","Kendall-Tau",kendall,0,x_axis)
            create_plot("Average Kendall-Tau with original list","plt/kt_orig_cmp.PNG","Epochs","Kendall-Tau",kendall,1,x_axis)
            create_plot("Average RBO measure with original list","plt/rbo_min_orig_cmp.PNG","Epochs","RBO",rbo_min,1,x_axis)
            create_plot("Average RBO measure with last iteration","plt/rbo_min_cmp.PNG","Epochs","RBO",rbo_min,0,x_axis)
            create_plot("Number of queries with winner changed", "plt/winner_change_cmp.PNG", "Epochs", "#Queries",cr,0, x_axis)"""

    def get_doc_names_clueWeb(self,clueweb_path):
        with open(clueweb_path) as features:
            name_index = {i:doc.split(" # ")[1].rstrip() for i,doc in enumerate(features)}
        with open(clueweb_path) as features:
            return name_index

    def create_trec_eval_file(self,score_file,names_index):
        trec = open("trec_Lambda_mart",'w')
        with open(score_file) as scores:
            scores_index = {i:score.split()[2].rstrip() for i,score in enumerate(scores)}
        with open(score_file) as scores:
            query_index = {i:score.split()[0].rstrip() for i,score in enumerate(scores)}

            for index in scores_index:
                line = query_index[index]+" Q0 "+names_index[index]+" 0 "+scores_index[index]+" seo\n"
                trec.write(line)
            trec.close()


    def get_score_for_LambdaMart_on_clueWeb(self,clueweb_path):
        name_index = self.get_doc_names_clueWeb(clueweb_path)
        score_file = self.run_lambda_mart(clueweb_path,0)
        self.create_trec_eval_file(score_file,name_index)


