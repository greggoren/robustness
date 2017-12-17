import itertools
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
    #return iter(p.stdout.readline,'')

class analyze:



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


    def extract_score(self, scores):
        for svm in scores:
            for epoch in scores[svm]:
                part = svm[1].split(".pickle")
                name = part[0]+part[1].replace(".","")+svm[2]
                f = open(name+str(epoch)+".txt",'w')
                for query in scores[svm][epoch]:
                    for doc in scores[svm][epoch][query]:
                        f.write(str(query).zfill(3)+" Q0 "+"ROUND-0"+str(epoch)+"-"+str(query).zfill(3)+"-"+doc+" "+str(scores[svm][epoch][query][doc]) +" "+ str(scores[svm][epoch][query][doc])+" seo\n")
                f.close()

    def calculate_metrics(self,models):
        metrics = {}
        for svm in models:
            ndcg_by_epochs = []
            map_by_epochs = []
            mrr_by_epochs = []
            for i in range(1,5):
                part = svm[1].split(".pickle")
                name = part[0] + part[1].replace(".", "")+svm[2]

                score_file = name+str(i)+".txt"
                qrels = "../rel/rel0"+str(i)
                command = "../trec_eval -q -m ndcg "+qrels+" "+score_file
                tmp=[]
                for line in run_command(command):
                    print(line)
                    if len(line.split())>1:
                        ndcg_score = line.split()[2].rstrip()
                        query =line.split()[1].rstrip()
                        if query!="all":
                            tmp.append(ndcg_score)
                    else:
                        break

                ndcg_by_epochs.append(np.median([float(a) for a in tmp]))

                command1 = "../trec_eval -q -m map " + qrels + " " + score_file
                tmp=[]
                for line in run_command(command1):
                    print(line)
                    if len(line.split()) > 1:
                        map_score = line.split()[2].rstrip()
                        query = line.split()[1].rstrip()
                        if query != "all":
                            tmp.append(map_score)
                    else:
                        break
                map_by_epochs.append(np.median([float(a) for a in tmp]))
                tmp=[]
                command2 = "../trec_eval -q -m recip_rank " + qrels + " " + score_file
                for line in run_command(command2):
                    print(line)
                    if len(line.split()) > 1:
                        mrr_score = line.split()[2].rstrip()
                        query = line.split()[1].rstrip()
                        if query != "all":
                            tmp.append(mrr_score)
                    else:
                        break
                mrr_by_epochs.append(np.median([float(a) for a in tmp]))

            metrics[svm] = (ndcg_by_epochs,map_by_epochs,mrr_by_epochs)
        return metrics

    def create_epsilon_for_Lambda_mart(self, competition_data, svm, banned_queries):
        scores = {}
        tmp = self.create_lambdaMart_scores(competition_data)
        tmp2 = self.get_all_scores(svm, competition_data)
        rankings = self.retrieve_ranking(scores)
        # epsilons = [0, 10, 20, 30, 40, 50, 60, 70,80,90,100]
        epsilons = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for epsilon in epsilons:
            key_lambdaMart = ("", "l.pickle1", "LambdaMart" + "_" + str(epsilon), "b")
            key_svm = ("", "l.pickle1", "SVMRank" + "_" + str(epsilon), "b")
            scores[key_lambdaMart] = tmp
            scores[key_svm] = tmp2[svm[0]]
            rankings[key_lambdaMart], scores = self.rerank_by_epsilon(key_lambdaMart, scores, epsilon, 1)
            rankings[key_svm], scores = self.rerank_by_epsilon(key_svm, scores, epsilon, 1)
        kendall, cr, rbo_min, x_axis, a = self.calculate_average_kendall_tau(rankings, [], banned_queries)
        self.extract_score(scores)
        metrics = self.calculate_metrics(scores)
        table_file = open("out/table_value_epsilons_LmbdaMart.tex", 'w')
        table_file.write("\\begin{longtable}{*{13}{c}}\n")
        table_file.write(
            "Ranker & Avg KT & Max KT & Avg RBO & Max RBO & WC & Min WC & Avg NDCG@5 & MAP & MRR  \\\\\\\\ \n")
        for key_lambdaMart in kendall:
            if key_lambdaMart[2].__contains__("SVMRank"):
                continue
            kt_avg = str(round(np.mean(kendall[key_lambdaMart][0]), 3))
            max_kt = str(round(max(kendall[key_lambdaMart][0]), 3))
            avg_rbo = str(round(np.mean(rbo_min[key_lambdaMart][0]), 3))
            max_rbo = str(round(max(rbo_min[key_lambdaMart][0]), 3))
            change = str(round(np.mean(cr[key_lambdaMart][0]), 3))
            m_change = str(round(min(cr[key_lambdaMart][0]), 3))
            nd = str(round(np.mean([float(a) for a in metrics[key_lambdaMart][0]]), 3))
            map = str(round(np.mean([float(a) for a in metrics[key_lambdaMart][1]]), 3))
            mrr = str(round(np.mean([float(a) for a in metrics[key_lambdaMart][2]]), 3))
            tmp = [kt_avg, max_kt, avg_rbo, max_rbo, change, m_change, nd, map, mrr]
            line = key_lambdaMart[2] + " & " + " & ".join(tmp) + " \\\\ \n"
            table_file.write(line)
            print(metrics[key_lambdaMart][2])
        table_file.write("\\end{longtable}")

    def calculate_average_kendall_tau(self, rankings,values,banned_queries):
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
                    # if query in banned_queries[epoch] or query in banned_queries[epoch-1]:
                    #     continue
                    current_list_svm = rankings_list_svm[epoch][query]
                    if not last_list_index_svm.get(query,False):
                        last_list_index_svm[query]=current_list_svm
                        original_list_index_svm[query]=current_list_svm
                        continue
                    if current_list_svm.index(5)!=last_list_index_svm[query].index(5):
                        # if  query not in banned_queries[epoch] and query not in banned_queries[epoch-1]:
                        change_rate_svm +=1
                    # if  query not in banned_queries[epoch] and query not in banned_queries[epoch - 1]:
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


    def get_all_scores(self,svms,competition_data):
        scores = {}
        for svm in svms:
            scores[svm] = {}
            epochs = range(1,5)
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
                if fixed[0] != retrieved_list_svm[0]:
                    new_scores[epoch][query]={fixed[0]:scores[svm][epoch][query][fixed[0]],fixed[1]:scores[svm][epoch-1][query][fixed[1]]}
                else:
                    new_scores[epoch][query] = scores[svm][epoch][query]

        scores[svm] = new_scores

        return rankings_svm[svm],scores

    def fix_ranking(self,svm,query,scores,epsilon,epoch,current_ranking,last_ranking,model):
        new_rank =[]
        if model==2:
            condorcet_count = {doc: 0 for doc in current_ranking}
            doc_pairs = list(itertools.combinations(current_ranking,2))
            for pair in doc_pairs:
                doc_win,doc_lose=self.determine_order(pair,current_ranking)
                if last_ranking.index(doc_lose) < last_ranking.index(doc_win) and (abs((scores[svm][epoch][query][doc_win]-scores[svm][epoch][query][doc_lose])/scores[svm][epoch][query][doc_lose])) < float(epsilon)/100:
                    # scores[svm][epoch][query][doc_win] - scores[svm][epoch][query][doc_lose]) < epsilon:

                    if (svm==("", "l.pickle1", "LambdaMart" + "_" + str(epsilon), "b")):
                        print(abs((scores[svm][epoch][query][doc_win]-scores[svm][epoch][query][doc_lose])/scores[svm][epoch][query][doc_lose]))
                    condorcet_count[doc_lose]+=1
                else:
                    if (svm==("", "l.pickle1", "LambdaMart" + "_" + str(epsilon), "b")):
                        print("score_change:",abs((scores[svm][epoch][query][doc_win]-scores[svm][epoch][query][doc_lose])/scores[svm][epoch][query][doc_lose]))
                        print("epsilon:",float(epsilon)/100)
                    condorcet_count[doc_win]+=1
            new_rank = sorted(current_ranking,key=lambda x:(condorcet_count[x],len(current_ranking)-current_ranking.index(x)),reverse = True)
        if model==3:
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

    def run_lambda_mart(self,features,epoch):
        java_path = "/lv_local/home/sgregory/jdk1.8.0_121/bin/java"
        jar_path = "/lv_local/home/sgregory/SEO_CODE/model_running/RankLib.jar"
        score_file = "score"+str(epoch)
        features= features
        model_path  = "/lv_local/home/sgregory/robustness/testmodel_250_50"
        command = java_path+" -jar "+jar_path + " -load "+model_path+" -rank "+features+ " -score "+score_file
        run_bash_command(command)
        return score_file

    def create_lambdaMart_scores(self,competition_data):
        scores={e :{} for e in competition_data}
        order = {_e: {} for _e in competition_data}
        for epoch in competition_data:
            scores[epoch]={q:{} for q in competition_data[epoch]}
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