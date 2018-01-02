import subprocess
import numpy as np
from scipy.stats import kendalltau
import RBO as r

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

    def calculate_metrics(self,models):
        metrics = {}
        for svm in models:
            ndcg_by_epochs = []
            map_by_epochs = []
            mrr_by_epochs = []
            for i in range(1, 9):
                name = "svm" + svm.split("svm_model")[1]

                score_file = name+str(i)
                qrels = "../rel/rel0" + str(i)

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

    def create_table(self, competition_data, svms, banned_queries):
        scores = self.get_all_scores(svms, competition_data)
        rankings = self.retrieve_ranking(scores)
        kendall, change_rate, rbo_min_models = self.calculate_average_kendall_tau(rankings, [])
        # self.extract_score(scores)
        # metrics = self.calculate_metrics(scores)

        table_file = open("table_value.tex", 'w')
        table_file.write("\\begin{longtable}{*{7}{c}}\n")
        table_file.write(
            "Ranker & C & Avg KT & Max KT & Avg RBO & Max RBO & WC & Min WC & NDCD & MAP & MRR \\\\\\\\ \n")
        for key in change_rate:
            model = key.split("svm_model")[1]
            average_kt = str(round(np.mean(kendall[key][0]), 3))
            max_kt = str(round(max(kendall[key][0]), 3))
            average_rbo = str(round(np.mean(rbo_min_models[key][0]), 3))
            max_rbo = str(round(max(rbo_min_models[key][0]), 3))
            change = str(round(np.mean(change_rate[key][0]), 3))
            m_change = str(round(min(change_rate[key][0]), 3))
            # nd = str(round(np.mean([float(a) for a in metrics[key][0]]), 3))
            # map = str(round(np.mean([float(a) for a in metrics[key][1]]), 3))
            # mrr = str(round(np.mean([float(a) for a in metrics[key][2]]), 3))
            tmp = ["SVMRank", model, average_kt, max_kt, average_rbo, max_rbo, change, m_change]  # , nd, map, mrr]
            line = " & ".join(tmp) + " \\\\ \n"
            table_file.write(line)
        table_file.write("\\end{longtable}")

    def calculate_average_kendall_tau(self, rankings, values):
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
                meta_rbo[svm] = {p: [] for p in values}
                for query in rankings_list_svm[epoch]:
                    current_list_svm = rankings_list_svm[epoch][query]
                    if not last_list_index_svm.get(query,False):
                        last_list_index_svm[query]=current_list_svm
                        original_list_index_svm[query]=current_list_svm
                        continue
                    # if current_list_svm.index(len(current_list_svm)) != last_list_index_svm[query].index(
                    if current_list_svm.index(5) != last_list_index_svm[query].index(
                            5):
                        change_rate_svm += 1
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
                change_rate_svm_epochs.append(float(change_rate_svm)/n_q)
                kt_svm.append(float(sum_svm) / n_q)
                kt_svm_orig.append(float(sum_svm_original) / n_q)
                rbo_min.append(float(sum_rbo_min) / n_q)
                rbo_min_orig.append(float(sum_rbo_min_orig) / n_q)
            kendall[svm] = (kt_svm, kt_svm_orig)
            rbo_min_models[svm] = (rbo_min, rbo_min_orig)
            change_rate[svm]=(change_rate_svm_epochs,)
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

