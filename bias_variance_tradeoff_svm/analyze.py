import subprocess
import numpy as np
import pickle
from scipy.stats import kendalltau
from kendall_tau import weighted_kendall_tau
import math
from statsmodels.genmod.families.links import sqrt
from sklearn.metrics.pairwise import cosine_similarity
import RBO as r
from scipy.stats import pearsonr
from scipy.stats import spearmanr
def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,shell=True
                         )
    return iter(p.stdout.readline,'')

def run_bash_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,shell=True)
    out,err = p.communicate()
    print(out)
    return out

class analyze:
    def order_trec_file(self,trec_file):
        final = trec_file.replace(".txt","")
        command = "sort -k1,1 -k5nr -k2,1 "+trec_file+" > "+final
        for line in run_bash_command(command):
            print(line)
        command = "rm "+trec_file
        for line in run_bash_command(command):
            print(line)
        return final

    def extract_score(self, scores):
        for svm in scores:
            for epoch in scores[svm]:
                name = "svm" + svm.split("svm_model")[1]
                f = open(name+str(epoch)+".txt",'w')
                for query in scores[svm][epoch]:
                    for doc in scores[svm][epoch][query]:
                        f.write(str(query).zfill(3)+" Q0 "+"ROUND-0"+str(epoch)+"-"+str(query).zfill(3)+"-"+doc+" "+str(scores[svm][epoch][query][doc]) +" "+ str(scores[svm][epoch][query][doc])+" seo\n")
                f.close()
                self.order_trec_file(name+str(epoch)+".txt")

    def cosine_similarity(self, v1, v2):
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i];
            y = v2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        return sumxy / math.sqrt(sumxx * sumyy)

    def calculate_metrics(self,models):
        metrics = {}
        for svm in models:
            ndcg_by_epochs = []
            map_by_epochs = []
            mrr_by_epochs = []
            for i in range(1, 9):
                name = "svm" + svm.split("svm_model")[1]

                score_file = name+str(i)
                qrels = "../rel3/rel0" + str(i)

                command = "../trec_eval -m ndcg "+qrels+" "+score_file
                for line in run_command(command):
                    print(line)
                    ndcg_score = line.split()[2].rstrip()
                    ndcg_by_epochs.append(ndcg_score)
                    break
                command1 = "../trec_eval -m map " + qrels + " " + score_file
                for line in run_command(command1):
                    print(line)
                    map_score = line.split()[2].rstrip()
                    map_by_epochs.append(map_score)
                    break
                command2 = "../trec_eval -m recip_rank " + qrels + " " + score_file
                for line in run_command(command2):
                    print(line)
                    mrr_score = line.split()[2].rstrip()
                    mrr_by_epochs.append(mrr_score)
                    break
            metrics[svm] = (ndcg_by_epochs,map_by_epochs,mrr_by_epochs)
        return metrics

    def create_change_percentage(self, cd):
        change = {}
        for epoch in cd:
            if epoch == 1:
                continue
            change[epoch] = {}
            for query in cd[epoch]:
                change[epoch][query] = {}
                for doc in cd[epoch][query]:
                    # change[epoch][query][doc] = np.linalg.norm(
                    #     cd[epoch][query][doc] - cd[epoch - 1][query][doc], ord=1) / np.linalg.norm(
                    #     cd[epoch - 1][query][doc], ord=1)
                    # change[epoch][query][doc] = float(1) / self.cosine_similarity(cd[epoch - 1][query][doc],
                    #                                                               cd[epoch][query][doc])
                    change[epoch][query][doc] = float(abs(np.linalg.norm(cd[epoch][query][doc]) - np.linalg.norm(
                        cd[epoch - 1][query][doc]))) / np.linalg.norm(cd[epoch - 1][query][doc])

        return change



    def create_table(self, competition_data, svms, banned_queries):

        weights = self.create_change_percentage(competition_data)
        scores = self.get_all_scores(svms, competition_data)
        rankings, ranks = self.retrieve_ranking(scores)

        kendall, change_rate, rbo_min_models = self.calculate_average_kendall_tau(rankings, weights, ranks)
        self.extract_score(scores)
        metrics = self.calculate_metrics(scores)

        table_file = open("table_value.tex", 'w')
        table_file.write("\\begin{longtable}{*{7}{c}}\n")
        table_file.write(
            "Ranker & C & Avg KT & Max KT & Avg RBO & Max RBO & WC & Min WC & NDCD & MAP & MRR \\\\\\\\ \n")
        keys = list(change_rate.keys())
        keys = sorted(keys, key=lambda x: float(x.split("svm_model")[1]))
        kendall_for_pearson = []
        kendall_max_for_pearson = []
        kendall_mean_for_pearson = []
        rbo_for_pearson = []
        wc_max_for_pearson = []
        wc_mean_for_pearson = []
        wc_pearson = []
        wc_weighted_for_pearson = []
        C_for_pearson = []
        ndcg_for_pearson = []
        map_for_pearson = []
        mrr_for_pearson = []
        wc_for_pearson = []
        wc_winner_for_pearson = []
        for key in keys:
            model = key.split("svm_model")[1]
            C_for_pearson.append(float(model))
            average_kt = str(round(np.mean(kendall[key][0]), 3))
            kendall_for_pearson.append(float(average_kt))
            average_rbo = str(round(np.mean(rbo_min_models[key][0]), 3))
            rbo_for_pearson.append(float(average_rbo))
            change_max = np.mean(change_rate[key][0])
            wc_max_for_pearson.append(change_max)
            change_mean = round(np.mean(change_rate[key][2]), 3)
            change = np.mean(change_rate[key][4])
            wc_for_pearson.append(change)
            winner_change_winner = np.mean(change_rate[key][3])
            wc_winner_for_pearson.append(winner_change_winner)
            wc_mean_for_pearson.append(change_mean)
            change_weighted = round(np.mean(change_rate[key][1]), 3)
            wc_weighted_for_pearson.append(change_weighted)
            nd = str(round(np.mean([float(a) for a in metrics[key][0]]), 3))
            ndcg_for_pearson.append(float(nd))
            map = str(round(np.mean([float(a) for a in metrics[key][1]]), 3))
            map_for_pearson.append(float(map))
            mrr = str(round(np.mean([float(a) for a in metrics[key][2]]), 3))
            mrr_for_pearson.append(float(mrr))
            max_w_kt = np.mean(kendall[key][2])
            kendall_max_for_pearson.append(max_w_kt)
            mean_w_kt = np.mean(kendall[key][3])
            kendall_mean_for_pearson.append(mean_w_kt)
        table_file.write("\\end{longtable}")

        f = open("spearman_correlation.tex", 'w')
        f.write("\\begin{tabular}{c|c|c|c} \n")
        f.write("Metric & #Tress & #Leaves \\\\ \n")
        corr_trees = spearmanr(C_for_pearson, kendall_for_pearson)
        f.write(
            "Kendall-$\\tau$ & " + str(round(corr_trees[0], 3)) + " (" + str(round(corr_trees[1], 3)) + ")   \\\\ \n")
        corr_trees = spearmanr(C_for_pearson, kendall_max_for_pearson)
        f.write("Kendall-$\\tau$ max normalized & " + str(round(corr_trees[0], 3)) + " (" + str(
            round(corr_trees[1], 3)) + ")  \\\\ \n")
        corr_trees = spearmanr(C_for_pearson, kendall_mean_for_pearson)
        f.write("Kendall-$\\tau$ mean normalized & " + str(round(corr_trees[0], 3)) + " (" + str(
            round(corr_trees[1], 3)) + ")  \\\\ \n")
        corr_trees = spearmanr(C_for_pearson, wc_mean_for_pearson)
        f.write(
            "Winner Change mean& " + str(round(corr_trees[0], 3)) + " (" + str(
                round(corr_trees[1], 3)) + ") &  \\\\ \n")
        corr_trees = spearmanr(C_for_pearson, wc_max_for_pearson)
        f.write(
            "Winner Change max & " + str(round(corr_trees[0], 3)) + " (" + str(round(corr_trees[1], 3)) + ") \\\\ \n")
        corr_trees = spearmanr(C_for_pearson, wc_mean_for_pearson)
        f.write(
            "Winner Change mean & " + str(round(corr_trees[0], 3)) + " (" + str(
                round(corr_trees[1], 3)) + ") &  \\\\ \n")
        corr_trees = spearmanr(C_for_pearson, wc_weighted_for_pearson)
        f.write("Winner Change weighted & " + str(round(corr_trees[0], 3)) + " (" + str(
            round(corr_trees[1], 3)) + ") \\\\ \n")
        corr_trees = spearmanr(C_for_pearson, wc_for_pearson)
        print(wc_for_pearson)
        f.write(
            "Winner Change & " + str(round(corr_trees[0], 3)) + " (" + str(round(corr_trees[1], 3)) + ") \\\\ \n")
        corr_trees = spearmanr(C_for_pearson, wc_winner_for_pearson)
        f.write("Winner Change winner & " + str(round(corr_trees[0], 3)) + " (" + str(
            round(corr_trees[1], 3)) + ")  \\\\ \n")
        corr_trees = spearmanr(C_for_pearson, rbo_for_pearson)
        f.write("RBO & " + str(round(corr_trees[0], 3)) + " (" + str(round(corr_trees[1], 3)) + ") \\\\ \n")
        f.write("\\end{tabular}")
        f.close()

    def calculate_average_kendall_tau(self, rankings, weights, ranks):
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
            change_rate_svm_epochs_max = []
            change_rate_svm_epochs_mean = []
            change_rate_svm_epochs_weighted = []
            change_rate_svm_epochs = []
            change_rate_winner_svm_epochs = []
            rbo_min = []
            rbo_min_orig = []
            max_w_kt_svm = []
            mean_w_kt_svm = []
            epochs = sorted(list(rankings_list_svm.keys()))
            for epoch in epochs:

                sum_svm = 0
                sum_rbo_min = 0
                sum_rbo_min_orig = 0
                sum_svm_original = 0
                sum_max_w_kt = 0
                sum_mean_w_kt = 0
                n_q=0
                change_rate_svm_mean = 0
                wc_change = 0
                change_rate_winner_svm = 0
                change_rate_svm_max = 0
                change_rate_svm_weighted = 0
                for query in rankings_list_svm[epoch]:
                    current_list_svm = rankings_list_svm[epoch][query]
                    if not last_list_index_svm.get(query,False):
                        last_list_index_svm[query]=current_list_svm
                        original_list_index_svm[query]=current_list_svm
                        continue
                    # if current_list_svm.index(len(current_list_svm)) != last_list_index_svm[query].index(
                    if current_list_svm.index(5) != last_list_index_svm[query].index(5):
                        wc_change += 1
                        change_rate_svm_max += (
                            float(1) / max([weights[epoch][query][ranks[svm][epoch][query][0]],
                                            weights[epoch][query][ranks[svm][epoch - 1][query][0]]]))
                        change_rate_svm_mean += (
                            float(1) / np.mean([weights[epoch][query][ranks[svm][epoch][query][0]],
                                                weights[epoch][query][ranks[svm][epoch - 1][query][0]]]))
                        change_rate_winner_svm += (
                            float(1) / (weights[epoch][query][ranks[svm][epoch][query][0]] + 1))
                        change_rate_svm_weighted += (
                            float(1) / (float(3) / 4 * weights[epoch][query][ranks[svm][epoch][query][0]] +
                                        weights[epoch][query][ranks[svm][epoch - 1][query][0]] * float(1) / 4))
                    sum_max_w_kt += weighted_kendall_tau(ranks[svm][epoch - 1][query], ranks[svm][epoch][query],
                                                         weights[epoch][query], "max")
                    sum_mean_w_kt += weighted_kendall_tau(ranks[svm][epoch - 1][query], ranks[svm][epoch][query],
                                                          weights[epoch][query], "mean")

                    n_q += 1
                    kt = kendalltau(last_list_index_svm[query], current_list_svm)[0]
                    kt_orig = kendalltau(original_list_index_svm[query], current_list_svm)[0]
                    rbo_orig = r.rbo_dict({x: j for x, j in enumerate(original_list_index_svm[query])},
                                          {x: j for x, j in enumerate(current_list_svm)}, 0.7)["min"]
                    rbo = r.rbo_dict({x: j for x, j in enumerate(last_list_index_svm[query])},
                                     {x: j for x, j in enumerate(current_list_svm)}, 0.7)["min"]
                    sum_rbo_min += rbo
                    sum_rbo_min_orig += rbo_orig
                    if not np.isnan(kt):
                        sum_svm += kt
                    if not np.isnan(kt_orig):
                        sum_svm_original += kt_orig
                    last_list_index_svm[query] = current_list_svm
                if n_q==0:
                    continue
                max_w_kt_svm.append(float(sum_max_w_kt) / n_q)
                mean_w_kt_svm.append(float(sum_mean_w_kt) / n_q)
                change_rate_winner_svm_epochs.append(float(change_rate_winner_svm) / n_q)
                change_rate_svm_epochs_max.append(float(change_rate_svm_max) / n_q)
                change_rate_svm_epochs.append(float(wc_change) / n_q)
                change_rate_svm_epochs_mean.append(float(change_rate_svm_mean) / n_q)
                change_rate_svm_epochs_weighted.append(float(change_rate_svm_weighted) / n_q)
                kt_svm.append(float(sum_svm) / n_q)
                kt_svm_orig.append(float(sum_svm_original) / n_q)
                rbo_min.append(float(sum_rbo_min) / n_q)
                rbo_min_orig.append(float(sum_rbo_min_orig) / n_q)
            kendall[svm] = (kt_svm, kt_svm_orig, max_w_kt_svm, mean_w_kt_svm)
            rbo_min_models[svm] = (rbo_min, rbo_min_orig)
            change_rate[svm] = (
                change_rate_svm_epochs_max, change_rate_svm_epochs_weighted, change_rate_svm_epochs_mean,
                change_rate_winner_svm_epochs, change_rate_svm_epochs)
        return kendall, change_rate, rbo_min_models


    def get_all_scores(self,svms,competition_data):
        scores = {}
        for svm in svms:

            scores[svm] = {}
            epochs = sorted(list(competition_data.keys()))
            print(epochs)
            for epoch in epochs:
                scores[svm][epoch] = {}
                for query in competition_data[epoch]:
                    scores[svm][epoch][query] = {}
                    for doc in competition_data[epoch][query]:
                        features_vector = competition_data[epoch][query][doc]
                        scores[svm][epoch][query][doc] = np.dot(svms[svm], features_vector.T)
        return scores

    def retrieve_ranking(self, scores):
        rankings_svm = {}
        optimized = False
        ranks = {}
        for svm in scores:
            ranks[svm] = {}
            if not optimized:
                competitors = self.get_competitors(scores[svm])
                optimized = True
            rankings_svm[svm] = {}
            scores_svm = scores[svm]
            for epoch in scores_svm:
                rankings_svm[svm][epoch] = {}
                ranks[svm][epoch] = {}
                for query in scores_svm[epoch]:
                    retrieved_list_svm = sorted(competitors[query], key=lambda x: (scores_svm[epoch][query][x], x),
                                                reverse=True)
                    rankings_svm[svm][epoch][query] = self.transition_to_rank_vector(competitors[query],
                                                                                     retrieved_list_svm)
                    ranks[svm][epoch][query] = retrieved_list_svm
        return rankings_svm, ranks

    def transition_to_rank_vector(self,original_list,sorted_list):
        rank_vector = []
        for doc in original_list:
            try:
                rank_vector.append(5 - (sorted_list.index(doc)))
            except:
                print(original_list,sorted_list)
        return rank_vector

    def get_competitors(self,scores_svm):
        competitors={}
        for query in scores_svm[1]:
            competitors[query] = scores_svm[1][query].keys()
        return competitors






    def retrieve_scores(self,score_file,order,epoch,result):
        with open(score_file,'r') as scores:
            index = 0
            for score in scores:
                value = float(score.split()[2])
                doc,query = tuple(order[epoch][index].split("@@@"))
                result[epoch][query][doc]=value
                index+=1
        return result




    def retrive_qrel(self,qrel_file):
        qrel={}
        with open(qrel_file) as qrels:
            for q in qrels:
                splited = q.split()
                name = splited[2]
                epoch = int(name.split("-")[1])
                query = name.split("-")[2]
                doc = name.split("-")[3]
                rel = splited[3].rstrip()
                if not qrel.get(epoch,False):
                    qrel[epoch] = {}
                if not qrel[epoch].get(query,False):
                    qrel[epoch][query] = {}
                qrel[epoch][query][doc]=rel
        return qrel

