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

                f = open(name + "_" + str(epoch) + ".txt", 'w')
                for query in scores[model][epoch]:
                    for doc in scores[model][epoch][query]:
                        f.write(str(query).zfill(3) + " Q0 " + "ROUND-0" + str(epoch) + "-" + str(query).zfill(
                            3) + "-" + doc + " " + str(scores[model][epoch][query][doc]) + " " + str(
                            scores[model][epoch][query][doc]) + " seo\n")
                f.close()
                self.order_trec_file(name + "_" + str(epoch) + ".txt")

    def calculate_metrics(self,models):
        metrics = {}
        for model in models:
            ndcg_by_epochs = []
            map_by_epochs = []
            mrr_by_epochs = []
            for i in range(1, 9):
                name = model.split("model_")[1]

                score_file = name + "_" + str(i)
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
            metrics[model] = (ndcg_by_epochs, map_by_epochs, mrr_by_epochs)
        return metrics

    def create_table(self, competition_data, models, banned_queries):
        weights = self.create_change_percentage(competition_data)
        scores = self.create_lambdaMart_scores(competition_data, models)
        rankings, ranks = self.retrieve_ranking(scores)
        kendall, change_rate, rbo_min_models = self.calculate_average_kendall_tau(rankings, banned_queries, weights,
                                                                                  ranks)
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
        wc_max_for_pearson = []
        wc_mean_for_pearson = []
        wc_weighted_for_pearson = []
        ndcg_for_pearson = []
        map_for_pearson = []
        mrr_for_pearson = []
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
            change_max = round(np.mean(change_rate[key][0]), 3)
            wc_max_for_pearson.append(change_max)
            change_mean = round(np.mean(change_rate[key][2]), 3)
            # wc_geo_mean_for_pearson.append(change_geo_mean)
            # change_geo_mean = str(round(np.mean(change_rate[key][3]), 3))
            wc_mean_for_pearson.append(change_mean)
            change_weighted = round(np.mean(change_rate[key][1]), 3)
            wc_weighted_for_pearson.append(change_weighted)
            m_change = str(round(min(change_rate[key][0]), 3))
            nd = str(round(np.mean([float(a) for a in metrics[key][0]]), 3))
            ndcg_for_pearson.append(float(nd))
            map = str(round(np.mean([float(a) for a in metrics[key][1]]), 3))
            map_for_pearson.append(float(map))
            mrr = str(round(np.mean([float(a) for a in metrics[key][2]]), 3))
            mrr_for_pearson.append(float(mrr))
            # tmp = ["LambdaMart", trees, leaves, average_kt, max_kt, average_rbo, max_rbo, change, m_change, nd, map,
            #        mrr]
            # tmp = ["LambdaMart", trees, leaves, change, m_change, nd, map, mrr]
            # line = " & ".join(tmp) + " \\\\ \n"
            # table_file.write(line)
            # print(metrics[key_lambdaMart][2])
        table_file.write("\\end{longtable}")
        print("leaves")
        print(pearsonr(leaves_for_pearson, kendall_for_pearson))
        print(pearsonr(leaves_for_pearson, rbo_for_pearson))
        # print(pearsonr(leaves_for_pearson, wc_for_pearson))
        print("max")
        print(pearsonr(leaves_for_pearson, wc_max_for_pearson))
        print("weighted")
        print(pearsonr(leaves_for_pearson, wc_weighted_for_pearson))
        print("mean")
        print(pearsonr(leaves_for_pearson, wc_mean_for_pearson))
        print(pearsonr(leaves_for_pearson, ndcg_for_pearson))
        print(pearsonr(leaves_for_pearson, map_for_pearson))
        print(pearsonr(leaves_for_pearson, mrr_for_pearson))
        print("trees")
        print(pearsonr(trees_for_pearson, kendall_for_pearson))
        print(pearsonr(trees_for_pearson, rbo_for_pearson))
        print("max")
        print(pearsonr(trees_for_pearson, wc_max_for_pearson))
        print("weighted")
        print(pearsonr(trees_for_pearson, wc_weighted_for_pearson))
        print("mean")
        print(pearsonr(trees_for_pearson, wc_mean_for_pearson))
        print(pearsonr(trees_for_pearson, ndcg_for_pearson))
        print(pearsonr(trees_for_pearson, map_for_pearson))
        print(pearsonr(trees_for_pearson, mrr_for_pearson))

    def create_change_percentage(self, cd):
        change = {}
        for epoch in cd:
            if epoch == 1:
                continue
            change[epoch] = {}
            for query in cd[epoch]:
                change[epoch][query] = {}
                for doc in cd[epoch][query]:
                    change[epoch][query][doc] = float(abs(np.linalg.norm(cd[epoch][query][doc]) - np.linalg.norm(
                        cd[epoch - 1][query][doc]))) / np.linalg.norm(cd[epoch - 1][query][doc])

        return change

    def calculate_average_kendall_tau(self, rankings, values, weights, ranks):
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
            rbo_min = []
            rbo_min_orig = []
            epochs = sorted(list(rankings_list_svm.keys()))
            for epoch in epochs:

                sum_svm = 0
                sum_rbo_min = 0
                sum_rbo_min_orig = 0
                sum_svm_original = 0
                n_q=0
                change_rate_svm_mean = 0
                # change_rate_svm_geo_mean = 0
                change_rate_svm_max = 0
                change_rate_svm_weighted = 0
                meta_rbo[svm] = {p: [] for p in values}
                for query in rankings_list_svm[epoch]:
                    current_list_svm = rankings_list_svm[epoch][query]
                    if not last_list_index_svm.get(query,False):
                        last_list_index_svm[query]=current_list_svm
                        original_list_index_svm[query]=current_list_svm
                        continue
                    # if current_list_svm.index(len(current_list_svm)) != last_list_index_svm[query].index(
                    if current_list_svm.index(5) != last_list_index_svm[query].index(5):
                        change_rate_svm_max += (
                            float(1) / max([weights[epoch][query][ranks[svm][epoch][query][0]],
                                            weights[epoch][query][ranks[svm][epoch - 1][query][0]]]))
                        change_rate_svm_mean += (
                            float(1) / np.mean([weights[epoch][query][ranks[svm][epoch][query][0]],
                                                weights[epoch][query][ranks[svm][epoch - 1][query][0]]]))
                        change_rate_svm_weighted += (
                            float(1) / (float(3) / 4 * weights[epoch][query][ranks[svm][epoch][query][0]] +
                                        weights[epoch][query][ranks[svm][epoch - 1][query][0]] * float(1) / 4))

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
                change_rate_svm_epochs_max.append(float(change_rate_svm_max) / n_q)
                change_rate_svm_epochs_mean.append(float(change_rate_svm_mean) / n_q)
                change_rate_svm_epochs_weighted.append(float(change_rate_svm_weighted) / n_q)
                kt_svm.append(float(sum_svm) / n_q)
                kt_svm_orig.append(float(sum_svm_original) / n_q)
                rbo_min.append(float(sum_rbo_min) / n_q)
                rbo_min_orig.append(float(sum_rbo_min_orig) / n_q)
            kendall[svm] = (kt_svm, kt_svm_orig)
            rbo_min_models[svm] = (rbo_min, rbo_min_orig)
            change_rate[svm] = (change_rate_svm_epochs_max, change_rate_svm_weighted, change_rate_svm_epochs_mean,)
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
        ranks = {}
        for svm in scores:
            ranks[svm] = {}
            if not optimized:
                competitors = self.get_competitors(scores[svm])
                optimized = True
            rankings_svm[svm]={}
            scores_svm = scores[svm]
            for epoch in scores_svm:
                rankings_svm[svm][epoch]={}
                ranks[svm][epoch] = {}
                for query in scores_svm[epoch]:
                    retrieved_list_svm = sorted(competitors[query],key=lambda x:(scores_svm[epoch][query][x],x),reverse=True)
                    rankings_svm[svm][epoch][query]= self.transition_to_rank_vector(competitors[query],retrieved_list_svm)
                    ranks[svm][epoch][query] = retrieved_list_svm
        return rankings_svm, ranks

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
