import itertools
import subprocess
import numpy as np
from scipy.stats import kendalltau
import RBO as r
from scipy.stats import pearsonr

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
    def create_lambdaMart_scores(self, competition_data, models):
        scores = {model: {epoch: {q: {} for q in list(competition_data[epoch].keys())} for epoch in competition_data}
                  for model in
                  models}
        print(scores)
        for epoch in competition_data:

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
            for model in models:
                score_file = self.run_lambda_mart(features_file, epoch, model)
                scores[model] = self.retrieve_scores(score_file, order, epoch, scores[model])
        return scores

    # s
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
        for model in scores:
            for epoch in scores[model]:
                name = model.split("model_")[1]

                f = open(name + "_" + str(epoch), 'w')
                for query in scores[model][epoch]:
                    for doc in scores[model][epoch][query]:
                        f.write(str(query).zfill(3) + " Q0 " + "ROUND-0" + str(epoch) + "-" + str(query).zfill(
                            3) + "-" + doc + " " + str(scores[model][epoch][query][doc]) + " " + str(
                            scores[model][epoch][query][doc]) + " seo\n")
                f.close()
                self.order_trec_file(name+str(epoch)+".txt")

    def calculate_metrics(self,models):
        metrics = {}
        for model in models:
            ndcg_by_epochs = []
            map_by_epochs = []
            mrr_by_epochs = []
            for i in range(1, 9):
                name = model.split("model_")[1]

                score_file = name + "_" + str(i)
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
            metrics[model] = (ndcg_by_epochs, map_by_epochs, mrr_by_epochs)
        return metrics

    def create_table(self, competition_data, models, banned_queries):
        scores = self.create_lambdaMart_scores(competition_data, models)
        rankings = self.retrieve_ranking(scores)
        kendall, change_rate, rbo_min_models = self.calculate_average_kendall_tau(rankings, banned_queries)
        self.extract_score(scores)
        metrics = self.calculate_metrics(scores)
        keys = list(change_rate.keys())
        keys = sorted(keys, key=lambda x: (
            float(x.split("model_")[1].split("_")[0]), float(x.split("model_")[1].split("_")[1])))  # TODO: fix split
        table_file = open("table_value_LmbdaMart.tex", 'w')
        table_file.write("\\begin{longtable}{*{12}{c}}\n")
        table_file.write(
            "Ranker & Trees & Leaves & Avg KT & Max KT & Avg RBO & Max RBO & WC & Min WC & NDCG & MAP & MRR  \\\\\\\\ \n")
        trees_for_pearson = []
        leaves_for_pearson = []
        kendall_for_pearson = []
        rbo_for_pearson = []
        wc_for_pearson = []
        for key in keys:
            trees, leaves = tuple((key.split("model_")[1].split("_")[0], key.split("model_")[1].split("_")[1]))
            trees_for_pearson.append(int(trees))
            leaves_for_pearson.append(int(leaves))
            average_kt = str(round(np.mean(kendall[key][0]), 3))
            kendall_for_pearson.append(float(average_kt))
            max_kt = str(round(max(kendall[key][0]), 3))
            average_rbo = str(round(np.mean(rbo_min_models[key][0]), 3))
            rbo_for_pearson.append(float(average_rbo))
            max_rbo = str(round(max(rbo_min_models[key][0]), 3))
            change = str(round(np.mean(change_rate[key][0]), 3))
            wc_for_pearson.append(float(change))
            m_change = str(round(min(change_rate[key][0]), 3))
            nd = str(round(np.mean([float(a) for a in metrics[key][0]]), 3))
            map = str(round(np.mean([float(a) for a in metrics[key][1]]), 3))
            mrr = str(round(np.mean([float(a) for a in metrics[key][2]]), 3))
            tmp = ["LambdaMart", trees, leaves, average_kt, max_kt, average_rbo, max_rbo, change, m_change, nd, map,
                   mrr]
            line = " & ".join(tmp) + " \\\\ \n"
            table_file.write(line)
            # print(metrics[key_lambdaMart][2])
        table_file.write("\\end{longtable}")
        print(pearsonr(trees_for_pearson, kendall_for_pearson))
        print(pearsonr(trees_for_pearson, rbo_for_pearson))
        print(pearsonr(trees_for_pearson, wc_for_pearson))

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
            change_rate_svm_epochs = []
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
                change_rate_svm_epochs.append(float(change_rate_svm) / n_q)
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
            for epoch in epochs:
                scores[svm][epoch] = {}
                for query in competition_data[epoch]:
                    scores[svm][epoch][query] = {}
                    fold = svm[0].query_to_fold_index[query]
                    weights_svm = svm[0].weights_index[fold]
                    for doc in competition_data[epoch][query]:
                        features_vector = competition_data[epoch][query][doc]
                        scores[svm][epoch][query][doc] = np.dot(weights_svm,features_vector.T )
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
                rank_vector.append(6-(sorted_list.index(doc)+1))
                # rank_vector.append(sorted_list.index(doc) + 1)
            except:
                print(original_list,sorted_list)
        return rank_vector

    def get_competitors(self,scores_svm):
        competitors={}
        for query in scores_svm[1]:
            competitors[query] = scores_svm[1][query].keys()
        return competitors




    def rerank_by_epsilon(self,svm,scores,epsilon,model):
        rankings_svm = {}
        ranked_docs={}
        new_scores ={}
        last_rank = {}
        competitors = self.get_competitors(scores[svm])
        rankings_svm[svm] = {}
        scores_svm = scores[svm]
        for epoch in scores_svm:
            ranked_docs[epoch]={}
            rankings_svm[svm][epoch] = {}
            new_scores[epoch] = {}
            for query in scores_svm[epoch]:

                retrieved_list_svm = sorted(competitors[query], key=lambda x: (scores_svm[epoch][query][x],x),
                                            reverse=True)

                if not last_rank.get(query,False):
                    last_rank[query] = retrieved_list_svm
                fixed = self.fix_ranking(svm,query,scores,epsilon,epoch,retrieved_list_svm,last_rank[query],model)
                ranked_docs[epoch][query] = fixed
                rankings_svm[svm][epoch][query] = self.transition_to_rank_vector(competitors[query],fixed)
                last_rank[query] = fixed
                if fixed[0] != retrieved_list_svm[0]:
                    new_scores[epoch][query]={fixed[0]:scores[svm][epoch][query][fixed[0]],fixed[1]:scores[svm][epoch-1][query][fixed[1]]}
                else:
                    new_scores[epoch][query] = scores[svm][epoch][query]
        scores[svm] = new_scores
        return rankings_svm[svm],scores,ranked_docs

    def determine_order(self,pair,current_ranking):
        if current_ranking.index(pair[0])<current_ranking.index(pair[1]):
            return pair[0],pair[1]
        else:
            return pair[1],pair[0]

    def fix_ranking(self,svm,query,scores,epsilon,epoch,current_ranking,last_ranking,model):
        new_rank =[]
        if model==0:
            for rank in range(len(current_ranking)):
                if rank + 1 < len(current_ranking):
                    if not new_rank:
                        doc_win =current_ranking[rank]
                    else:
                        doc_win=new_rank[rank]

                    doc_lose = current_ranking[rank+1]
                    if doc_win in new_rank:
                        new_rank = new_rank[:-1]
                # if last_ranking.index(doc_lose) < last_ranking.index(doc_win) and (abs((scores[svm][epoch][query][doc_win] - scores[svm][epoch][query][doc_lose])/scores[svm][epoch][query][doc_lose])) < float(epsilon) / 100:
                if last_ranking.index(doc_lose) < last_ranking.index(doc_win) and (
                    scores[svm][epoch][query][doc_win] - scores[svm][epoch][query][doc_lose]) < epsilon:
                    new_rank.append(doc_lose)
                    new_rank.append(doc_win)
                else:
                    new_rank.append(doc_win)
                    new_rank.append(doc_lose)
        if model==1:
            condorcet_count = {doc: 0 for doc in current_ranking}
            doc_pairs = list(itertools.combinations(current_ranking,2))
            for pair in doc_pairs:
                doc_win,doc_lose=self.determine_order(pair,current_ranking)
                if last_ranking.index(doc_lose) < last_ranking.index(doc_win) and (abs((scores[svm][epoch][query][doc_win]-scores[svm][epoch][query][doc_lose])/scores[svm][epoch][query][doc_lose])) < float(epsilon)/100:
                    # scores[svm][epoch][query][doc_win] - scores[svm][epoch][query][doc_lose]) < epsilon:
                    condorcet_count[doc_lose]+=1
                else:
                    condorcet_count[doc_win]+=1
            new_rank = sorted(current_ranking,key=lambda x:(condorcet_count[x],len(current_ranking)-current_ranking.index(x)),reverse = True)
        if model==2:
            last_winner = last_ranking[0]
            current_winner = current_ranking[0]

            if last_ranking.index(last_winner) < last_ranking.index(current_winner) and (abs(
                    (scores[svm][epoch][query][current_winner] - scores[svm][epoch][query][last_winner]) /
                    scores[svm][epoch][query][last_winner])) < float(epsilon) / 100:
                new_rank.append(last_winner)
                new_rank.append(current_winner)
                new_rank.extend([c for c in current_ranking if c!=last_winner and c!=current_winner])
            else:
                new_rank=current_ranking
        return new_rank

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

    def run_lambda_mart(self, features, epoch, model):
        java_path = "/lv_local/home/sgregory/jdk1.8.0_121/bin/java"
        jar_path = "/lv_local/home/sgregory/SEO_CODE/model_running/RankLib.jar"
        score_file = "score"+str(epoch)
        features= features
        model_path = model
        command = java_path+" -jar "+jar_path + " -load "+model_path+" -rank "+features+ " -score "+score_file
        run_bash_command(command)
        return score_file



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

    def mrr(self,qrel,rankings):
        mrr_for_ranker = {}
        for ranker in rankings:
            mrr_by_epochs =[]
            for epoch in rankings[ranker]:
                mrr=0
                nq=0
                for query in rankings[ranker][epoch]:
                    if query.__contains__("_2"):
                        continue
                    nq+=1
                    ranking_list = rankings[ranker][epoch][query]
                    try:
                        for doc in ranking_list:
                            if qrel[epoch][query][doc]!="0":
                                mrr+=(1.0/(ranking_list.index(doc)+1))
                                break
                    except:

                        print(qrel.keys())
                        print(epoch)
                        print(query)
                        print(doc)
                mrr_by_epochs.append(mrr/nq)
            mrr_for_ranker[ranker]=mrr_by_epochs
        return mrr_for_ranker
