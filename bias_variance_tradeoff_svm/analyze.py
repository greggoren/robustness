import subprocess
import numpy as np
import pickle
from kendall_tau import kendall_distance, weighted_kendall_distance, normalized_weighted_kendall_distance
import math
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

        for model in models:
            per_query_stats = {"ndcg": {}, "map": {}, "mrr": {}}
            queries = []
            metrics[model] = {}
            # per_query_stats[model]={}
            for i in range(1, 6):
                name = "svm" + model.split("_model")[1]

                score_file = name + str(i)
                qrels = "../rel2/srel0" + str(i)

                command = "../trec_eval -q -m ndcg " + qrels + " " + score_file
                for line in run_command(command):
                    if len(line.split()) <= 1:
                        break
                    if line.split()[1] == "all":
                        break
                    if not per_query_stats["ndcg"].get(i, False):
                        per_query_stats["ndcg"][i] = {}
                    ndcg_score = float(line.split()[2].rstrip())
                    query = line.split()[1]
                    queries.append(query)
                    queries = list(set(queries))
                    per_query_stats["ndcg"][i][query] = ndcg_score
                    # print(line)
                    # ndcg_score = line.split()[2].rstrip()
                    # ndcg_by_epochs.append(ndcg_score)
                command1 = "../trec_eval -q -m map " + qrels + " " + score_file
                for line in run_command(command1):
                    if len(line.split()) <= 1:
                        break
                    if line.split()[1] == "all":
                        break
                    if not per_query_stats["map"].get(i, False):
                        per_query_stats["map"][i] = {}
                    map_score = float(line.split()[2].rstrip())
                    query = line.split()[1]
                    per_query_stats["map"][i][query] = map_score
                    # print(line)
                    # map_score = line.split()[2].rstrip()
                    # map_by_epochs.append(map_score)
                    # break
                command2 = "../trec_eval -q -m recip_rank " + qrels + " " + score_file
                for line in run_command(command2):
                    if len(line.split()) <= 1:
                        break
                    if line.split()[1] == "all":
                        break
                    if not per_query_stats["mrr"].get(i, False):
                        per_query_stats["mrr"][i] = {}
                    mrr_score = float(line.split()[2].rstrip())
                    query = line.split()[1]
                    per_query_stats["mrr"][i][query] = mrr_score
                    # print(line)
                    # mrr_score = line.split()[2].rstrip()
                    # mrr_by_epochs.append(mrr_score)
                    # break
            # metrics[model] = (ndcg_by_epochs, map_by_epochs, mrr_by_epochs)

            averaged_rel_stats = self.average_metrics_for_queries_rel(per_query_stats, queries)
            f = open("query_rel_stats", 'wb')
            pickle.dump(averaged_rel_stats, f)
            f.close()
            ndcg_by_queries = [averaged_rel_stats["ndcg"][q] for q in averaged_rel_stats["ndcg"]]
            map_by_queries = [averaged_rel_stats["map"][q] for q in averaged_rel_stats["map"]]
            mrr_by_queries = [averaged_rel_stats["mrr"][q] for q in averaged_rel_stats["mrr"]]
            metrics[model] = (ndcg_by_queries, map_by_queries, mrr_by_queries)
        return metrics

    def average_metrics_for_queries_rel(self, metrics, queries):
        averaged_epochs = {m: {} for m in metrics}
        for metric in metrics:
            for query in queries:
                averaged_epochs[metric][query] = np.mean([metrics[metric][e][query] for e in metrics[metric]])
        return averaged_epochs

    def cosine_similarity(self, v1, v2):
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]
            y = v2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        return sumxy / math.sqrt(sumxx * sumyy)
    def create_change_percentage(self, cd):
        change = {}
        for epoch in cd:
            if epoch == 1:
                continue
            change[epoch] = {}
            for query in cd[epoch]:
                change[epoch][query] = {}
                for doc in cd[epoch][query]:
                    change[epoch][query][doc] = cd[epoch][query][doc] - cd[epoch - 1][query][doc]
                    # v1 = cd[epoch][query][doc] / np.linalg.norm(cd[epoch][query][doc])
                    # v2 = cd[epoch - 1][query][doc] / np.linalg.norm(cd[epoch - 1][query][doc])
                    # change[epoch][query][doc] = 1 - self.cosine_similarity(v1, v2)
        return change

    def normalzaied_metric_enforcer(self, metric, w1, w2, d1, d2):
        if metric == "diff":
            v1 = np.linalg.norm(w1) / np.linalg.norm(d1)
            v2 = np.linalg.norm(w2) / np.linalg.norm(d2)
            return abs(v2 - v1)
        if metric == "rel":
            return np.linalg.norm(w2 - w1) / np.linalg.norm(d2 - d1)
        if metric == "sum":
            v1 = np.linalg.norm(w2) / np.linalg.norm(d2)
            v2 = np.linalg.norm(w2) / np.linalg.norm(d2)
            return v1 + v2

    def determine_normzalize(self, w1, w2, metric):
        if metric == "diff":
            v1 = np.linalg.norm(w1)
            v2 = np.linalg.norm(w2)
            return abs(v2 - v1)
        if metric == "rel":
            return np.linalg.norm(w2 - w1)
        if metric == "sum":
            v1 = np.linalg.norm(w1)
            v2 = np.linalg.norm(w2)
            return v1 + v2

    def get_weighted_winner_change_score(self, former_winner, new_winner, weights, metric):
        value = float(1) / (1 + self.determine_normzalize(weights[former_winner], weights[new_winner], metric))
        return value

    def get_normalized_weighted_winner_change_score(self, former_winner, new_winner, weights, doc_fwinner, doc_nwinner,
                                                    metric):
        value = float(1) / (
            1 + self.normalzaied_metric_enforcer(metric, weights[former_winner], weights[new_winner], doc_fwinner,
                                                 doc_nwinner))
        return value

    def score_experiment(self, cd, models):
        scores = self.get_all_scores(models, cd)
        rankings, ranks = self.retrieve_ranking(scores)
        model_scores_diff_consecutive_winner_to_loser, model_scores_diff_current_former_winner = self.create_score_diffs(
            ranks, scores)
        sorted_models = sorted(list(models.keys()), key=lambda x: float(x.split("svm_model")[1]))
        C_for_corr = []
        cons_for_corr = []
        swap_for_corr = []
        for model in sorted_models:
            C = float(model.split("svm_model")[1])
            C_for_corr.append(C)
            cons_for_corr.append(model_scores_diff_consecutive_winner_to_loser[model])
            swap_for_corr.append(model_scores_diff_current_former_winner[model])
        print(pearsonr(C_for_corr, cons_for_corr))
        print(spearmanr(C_for_corr, cons_for_corr))
        print(pearsonr(C_for_corr, swap_for_corr))
        print(spearmanr(C_for_corr, swap_for_corr))

    def create_score_diffs(self, ranks, scores):
        model_scores_diff_current_former_winner = {}
        model_scores_diff_consecutive_winner_to_loser = {}
        for model in ranks:
            model_scores_diff_current_former_winner[model] = {}
            model_scores_diff_consecutive_winner_to_loser[model] = {}
            for epoch in ranks[model]:
                if epoch == 1:

                    continue
                model_scores_diff_current_former_winner[model][epoch] = []
                model_scores_diff_consecutive_winner_to_loser[model][epoch] = []
                for query in ranks[model][epoch]:
                    if ranks[model][epoch][query][0] != ranks[model][epoch - 1][query][0]:
                        former_winner = ranks[model][epoch - 1][query][0]
                        current_winner = ranks[model][epoch][query][0]
                        model_scores_diff_current_former_winner[model][epoch].append(float(abs(
                            scores[model][epoch][query][current_winner] - scores[model][epoch][query][
                                former_winner])) / abs(
                            scores[model][epoch - 1][query][current_winner] - scores[model][epoch - 1][query][
                                former_winner]))
                    else:
                        current_winner = ranks[model][epoch][query][0]
                        second = ranks[model][epoch][query][1]
                        value = float(abs(
                            scores[model][epoch][query][current_winner] - scores[model][epoch][query][second])) / abs(
                            scores[model][epoch - 1][query][current_winner] - scores[model][epoch - 1][query][second])
                        model_scores_diff_consecutive_winner_to_loser[model][epoch].append(value)
        for model in model_scores_diff_consecutive_winner_to_loser:
            for epoch in model_scores_diff_consecutive_winner_to_loser[model]:
                model_scores_diff_consecutive_winner_to_loser[model][epoch] = np.mean(
                    model_scores_diff_consecutive_winner_to_loser[model][epoch])
                model_scores_diff_current_former_winner[model][epoch] = np.mean(
                    model_scores_diff_current_former_winner[model][epoch])
        for model in model_scores_diff_current_former_winner:
            model_scores_diff_current_former_winner[model] = np.mean(
                [model_scores_diff_current_former_winner[model][e] for e in
                 model_scores_diff_current_former_winner[model]])
            model_scores_diff_consecutive_winner_to_loser[model] = np.mean(
                [model_scores_diff_consecutive_winner_to_loser[model][e] for e in
                 model_scores_diff_consecutive_winner_to_loser[model]])
        return model_scores_diff_consecutive_winner_to_loser, model_scores_diff_current_former_winner

    def create_table(self, competition_data, svms, banned_queries):

        weights = self.create_change_percentage(competition_data)
        scores = self.get_all_scores(svms, competition_data)
        rankings, ranks = self.retrieve_ranking(scores)

        kendall, change_rate, rbo_min_models = self.calculate_average_kendall_tau(rankings, weights, ranks,
                                                                                  banned_queries, competition_data)
        self.extract_score(scores)
        metrics = self.calculate_metrics(scores)

        table_file = open("table_value.tex", 'w')
        table_file.write(
            " Ranker \KTshort & \WC & \RBO & \WCdiff & \WCresp  & \WCsum & \KTDdiff & \KTDresp & \KTDsum & WC diff norm & WC rel norm  & WC sum norm & KTD diff norm & KTD rel norm & KTD sum norm \n")
        keys = list(change_rate.keys())
        keys = sorted(keys, key=lambda x: float(x.split("svm_model")[1]))
        kendall_for_pearson = []
        rbo_for_pearson = []
        wc_sum_for_pearson = []
        wc_sum_n_for_pearson = []
        wc_diff_for_pearson = []
        wc_diff_n_for_pearson = []
        wc_rel_for_pearson = []
        wc_rel_n_for_pearson = []
        ndcg_for_pearson = []
        map_for_pearson = []
        mrr_for_pearson = []
        kendall_sum_for_pearson = []
        kendall_sum_n_for_pearson = []
        kendall_diff_for_pearson = []
        kendall_diff_n_for_pearson = []
        kendall_rel_for_pearson = []
        kendall_rel_n_for_pearson = []
        wc_for_pearson = []
        rmetrics = []
        C_for_pearson = []

        for key in keys:
            model = key.split("svm_model")[1]
            C_for_pearson.append(float(model))
            average_kt = np.mean(kendall[key][0])
            kendall_for_pearson.append(float(average_kt))
            rel_kt = np.mean(kendall[key][5])
            kendall_rel_for_pearson.append(rel_kt)
            rel_kt_n = np.mean(kendall[key][6])
            kendall_rel_n_for_pearson.append(rel_kt_n)
            sum_kt = np.mean(kendall[key][1])
            kendall_sum_for_pearson.append(sum_kt)
            sum_kt_n = np.mean(kendall[key][2])
            kendall_sum_n_for_pearson.append(sum_kt_n)
            diff_kt = np.mean(kendall[key][3])
            kendall_diff_for_pearson.append(diff_kt)
            diff_kt_n = np.mean(kendall[key][4])
            kendall_diff_n_for_pearson.append(diff_kt_n)
            change_sum = np.mean(change_rate[key][0])
            wc_sum_for_pearson.append(change_sum)
            change_sum_n = np.mean(change_rate[key][1])
            wc_sum_n_for_pearson.append(change_sum_n)
            change_dif = np.mean(change_rate[key][2])
            wc_diff_for_pearson.append(change_dif)
            change_dif_n = np.mean(change_rate[key][3])
            wc_diff_n_for_pearson.append(change_dif_n)
            change = np.mean(change_rate[key][6])
            wc_for_pearson.append(change)
            change_rel = np.mean(change_rate[key][4])
            wc_rel_for_pearson.append(change_rel)
            change_rel_n = np.mean(change_rate[key][5])
            wc_rel_n_for_pearson.append(change_rel_n)
            average_rbo = np.mean(rbo_min_models[key][0])
            rbo_for_pearson.append(float(average_rbo))
            nd = np.mean([float(a) for a in metrics[key][0]])
            ndcg_for_pearson.append(float(nd))
            map = np.mean([float(a) for a in metrics[key][1]])
            map_for_pearson.append(float(map))
            mrr = np.mean([float(a) for a in metrics[key][2]])
            mrr_for_pearson.append(float(mrr))
            rmetrics.append(average_kt)
            rmetrics.append(change)
            rmetrics.append(average_rbo)
            rmetrics.append(change_dif)
            rmetrics.append(change_rel)
            rmetrics.append(change_sum)
            rmetrics.append(diff_kt)
            rmetrics.append(rel_kt)
            rmetrics.append(sum_kt)
            rmetrics.append(change_dif_n)
            rmetrics.append(change_rel_n)
            rmetrics.append(change_sum_n)
            rmetrics.append(diff_kt_n)
            rmetrics.append(rel_kt_n)
            rmetrics.append(sum_kt_n)
            rmetrics.append(nd)
            rmetrics.append(map)
            rmetrics.append(mrr)
            line = "SVMRank & " + " & ".join(["$" + str(round(a, 3)) + "$" for a in rmetrics])
            table_file.write(line)
        table_file.close()
        f = open("pearson_correlation.tex", 'w')
        f.write("\\begin{tabular}{c|c|c|c} \n")
        f.write("Metric & #Tress & #Leaves \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, kendall_for_pearson)
        f.write(
            "\KTshort & " + str(round(corr_trees[0], 3)) + " (" + str(round(corr_trees[1], 3)) + ")   \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, wc_for_pearson)
        f.write("\WC & " + str(round(corr_trees[0], 3)) + " (" + str(
            round(corr_trees[1], 3)) + ")  \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, rbo_for_pearson)
        f.write("\RBO & " + str(round(corr_trees[0], 3)) + " (" + str(
            round(corr_trees[1], 3)) + ")  \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, wc_diff_for_pearson)
        f.write(
            "\WCdiff & " + str(round(corr_trees[0], 3)) + " (" + str(
                round(corr_trees[1], 3)) + ") &  \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, wc_rel_for_pearson)
        f.write(
            "\WCresp & " + str(round(corr_trees[0], 3)) + " (" + str(round(corr_trees[1], 3)) + ") \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, wc_sum_for_pearson)
        f.write(
            "\WCsum & " + str(round(corr_trees[0], 3)) + " (" + str(
                round(corr_trees[1], 3)) + ") &  \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, kendall_diff_for_pearson)
        f.write("\KTDdiff & " + str(round(corr_trees[0], 3)) + " (" + str(
            round(corr_trees[1], 3)) + ") \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, kendall_rel_for_pearson)
        print(wc_for_pearson)
        f.write(
            "\KTDresp & " + str(round(corr_trees[0], 3)) + " (" + str(round(corr_trees[1], 3)) + ") \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, kendall_sum_for_pearson)
        f.write("\KTDsum & " + str(round(corr_trees[0], 3)) + " (" + str(
            round(corr_trees[1], 3)) + ")  \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, wc_diff_n_for_pearson)
        f.write("WC diff norm & " + str(round(corr_trees[0], 3)) + " (" + str(round(corr_trees[1], 3)) + ") \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, wc_rel_n_for_pearson)
        f.write("WC rel norm & " + str(round(corr_trees[0], 3)) + " (" + str(round(corr_trees[1], 3)) + ") \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, wc_sum_n_for_pearson)
        f.write("WC sum norm & " + str(round(corr_trees[0], 3)) + " (" + str(round(corr_trees[1], 3)) + ") \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, kendall_diff_n_for_pearson)
        f.write("KTD diff norm & " + str(round(corr_trees[0], 3)) + " (" + str(round(corr_trees[1], 3)) + ") \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, kendall_rel_n_for_pearson)
        f.write("KTD rel norm & " + str(round(corr_trees[0], 3)) + " (" + str(round(corr_trees[1], 3)) + ") \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, kendall_sum_n_for_pearson)
        f.write("KTD sum norm & " + str(round(corr_trees[0], 3)) + " (" + str(round(corr_trees[1], 3)) + ") \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, ndcg_for_pearson)
        f.write("\\ndcg & " + str(round(corr_trees[0], 3)) + " (" + str(round(corr_trees[1], 3)) + ") \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, map_for_pearson)
        f.write("\map & " + str(round(corr_trees[0], 3)) + " (" + str(round(corr_trees[1], 3)) + ") \\\\ \n")
        corr_trees = pearsonr(C_for_pearson, mrr_for_pearson)
        f.write("\mrr & " + str(round(corr_trees[0], 3)) + " (" + str(round(corr_trees[1], 3)) + ") \\\\ \n")
        f.write("\\end{tabular}")
        f.close()
        print("spearman")
        print("kt")
        print(spearmanr(C_for_pearson, kendall_for_pearson))
        print("wc")
        print(spearmanr(C_for_pearson, wc_for_pearson))
        print("pearson")
        print("kt")
        print(pearsonr(C_for_pearson, kendall_for_pearson))
        print("wc")
        print(pearsonr(C_for_pearson, wc_for_pearson))
        a = open("wc", 'wb')
        pickle.dump((C_for_pearson, wc_for_pearson), a)
        a.close()
        a = open("kt", 'wb')
        pickle.dump((C_for_pearson, kendall_for_pearson), a)
        a.close()

    def calculate_average_kendall_tau(self, rankings, weights, ranks, banned, cd):
        kendall = {}
        change_rate = {}
        rbo_min_models = {}

        for svm in rankings:
            queries = []
            rankings_list_svm = rankings[svm]
            kt_svm = []
            last_list_index_svm = {}
            original_list_index_svm = {}
            change_rate_sum = []
            change_rate_sum_n = []
            change_rate_rel = []
            change_rate_rel_n = []
            change_rate_diff = []
            change_rate_diff_n = []
            change_rate_svm_epochs = []
            rbo_min = []
            rbo_min_orig = []
            sum_kt_svm = []
            sum_kt_svm_n = []
            diff_kt_svm = []
            diff_kt_svm_n = []
            rel_kt_svm = []
            rel_kt_svm_n = []
            epochs = sorted(list(rankings_list_svm.keys()))
            metrics_for_stats = {"ktd": {e: {} for e in epochs}, "wc": {e: {} for e in epochs},
                                 "rbo": {e: {} for e in epochs}, "wc_diff": {e: {} for e in epochs},
                                 "wc_sum": {e: {} for e in epochs}, "wc_rel": {e: {} for e in epochs},
                                 "ktd_rel": {e: {} for e in epochs}, "ktd_diff": {e: {} for e in epochs},
                                 "ktd_sum": {e: {} for e in epochs}, "wc_n_diff": {e: {} for e in epochs},
                                 "wc_n_sum": {e: {} for e in epochs}, "wc_n_rel": {e: {} for e in epochs},
                                 "ktd_n_diff": {e: {} for e in epochs}, "ktd_n_sum": {e: {} for e in epochs},
                                 "ktd_n_rel": {e: {} for e in epochs}}

            for epoch in epochs:

                sum_svm = 0
                sum_rbo_min = 0
                sum_rbo_min_orig = 0
                sum_diff_kt = 0
                sum_diff_kt_n = 0
                sum_rel_kt = 0
                sum_rel_kt_n = 0
                sum_sum_kt = 0
                sum_sum_kt_n = 0
                n_q = 0
                wc_change = 0
                sum_change_rate_diff = 0
                sum_change_rate_diff_n = 0
                sum_change_rate_rel = 0
                sum_change_rate_rel_n = 0
                sum_change_rate_sum = 0
                sum_change_rate_sum_n = 0
                for query in rankings_list_svm[epoch]:
                    queries.append(query)
                    queries = list(set(queries))
                    current_list_svm = rankings_list_svm[epoch][query]
                    if not last_list_index_svm.get(query, False):
                        last_list_index_svm[query] = current_list_svm
                        original_list_index_svm[query] = current_list_svm
                        continue
                    if query not in banned[epoch] and query not in banned[epoch - 1]:

                        if ranks[svm][epoch][query][0] != ranks[svm][epoch - 1][query][0]:
                            wc = 1
                            wc_sum = self.get_weighted_winner_change_score(ranks[svm][epoch - 1][query][0],
                                                                           ranks[svm][epoch][query][0],
                                                                           weights[epoch][query], "sum")
                            wc_rel = self.get_weighted_winner_change_score(ranks[svm][epoch - 1][query][0],
                                                                           ranks[svm][epoch][query][0],
                                                                           weights[epoch][query], "rel")
                            wc_diff = self.get_weighted_winner_change_score(ranks[svm][epoch - 1][query][0],
                                                                            ranks[svm][epoch][query][0],
                                                                            weights[epoch][query], "diff")
                            wc_diff_n = self.get_normalized_weighted_winner_change_score(
                                ranks[svm][epoch - 1][query][0],
                                ranks[svm][epoch][query][0],
                                weights[epoch][query], cd[epoch - 1][query][ranks[svm][epoch - 1][query][0]],
                                cd[epoch - 1][query][ranks[svm][epoch][query][0]], "diff")

                            wc_sum_n = self.get_normalized_weighted_winner_change_score(
                                ranks[svm][epoch - 1][query][0],
                                ranks[svm][epoch][query][0],
                                weights[epoch][query], cd[epoch - 1][query][ranks[svm][epoch - 1][query][0]],
                                cd[epoch - 1][query][ranks[svm][epoch][query][0]], "sum")

                            wc_rel_n = self.get_normalized_weighted_winner_change_score(
                                ranks[svm][epoch - 1][query][0],
                                ranks[svm][epoch][query][0],
                                weights[epoch][query], cd[epoch - 1][query][ranks[svm][epoch - 1][query][0]],
                                cd[epoch - 1][query][ranks[svm][epoch][query][0]], "rel")

                            sum_change_rate_diff_n += wc_diff_n
                            sum_change_rate_sum_n += wc_sum_n
                            sum_change_rate_rel_n += wc_rel_n
                            sum_change_rate_diff += wc_diff
                            sum_change_rate_rel += wc_rel
                            sum_change_rate_sum += wc_sum
                            wc_change += wc
                        else:
                            wc = 0
                            wc_diff, wc_diff_n, wc_rel, wc_rel_n, wc_sum, wc_sum_n = 0, 0, 0, 0, 0, 0
                        diff_kt = weighted_kendall_distance(ranks[svm][epoch - 1][query],
                                                            ranks[svm][epoch][query],
                                                            weights[epoch][query], "diff")
                        sum_diff_kt += diff_kt
                        diff_kt_n = normalized_weighted_kendall_distance(ranks[svm][epoch - 1][query],
                                                                         ranks[svm][epoch][query],
                                                                         weights[epoch][query],
                                                                         cd[epoch - 1][query],
                                                                         "diff")
                        sum_diff_kt_n += diff_kt_n
                        sum_kt = weighted_kendall_distance(ranks[svm][epoch - 1][query],
                                                           ranks[svm][epoch][query],
                                                           weights[epoch][query], "sum")
                        sum_sum_kt += sum_kt
                        sum_kt_n = normalized_weighted_kendall_distance(ranks[svm][epoch - 1][query],
                                                                        ranks[svm][epoch][query],
                                                                        weights[epoch][query],
                                                                        cd[epoch - 1][query],
                                                                        "sum")
                        sum_sum_kt_n += sum_kt_n
                        rel_kt = weighted_kendall_distance(ranks[svm][epoch - 1][query],
                                                           ranks[svm][epoch][query],
                                                           weights[epoch][query], "rel")
                        sum_rel_kt += rel_kt
                        rel_kt_n = normalized_weighted_kendall_distance(ranks[svm][epoch - 1][query],
                                                                        ranks[svm][epoch][query],
                                                                        weights[epoch][query],
                                                                        cd[epoch - 1][query],
                                                                        "rel")

                        sum_rel_kt_n += rel_kt_n
                        n_q += 1
                        kt = kendall_distance(ranks[svm][epoch - 1][query], ranks[svm][epoch][query])
                        rbo_orig = r.rbo_dict({x: j for x, j in enumerate(original_list_index_svm[query])},
                                              {x: j for x, j in enumerate(current_list_svm)}, 0.7)["min"]
                        rbo = r.rbo_dict({x: j for x, j in enumerate(last_list_index_svm[query])},
                                         {x: j for x, j in enumerate(current_list_svm)}, 0.7)["min"]
                        sum_rbo_min += rbo
                        sum_rbo_min_orig += rbo_orig
                        if not np.isnan(kt):
                            sum_svm += kt
                        metrics_for_stats["ktd"][epoch][query] = kt
                        metrics_for_stats["ktd_diff"][epoch][query] = diff_kt
                        metrics_for_stats["ktd_n_diff"][epoch][query] = diff_kt_n
                        metrics_for_stats["ktd_sum"][epoch][query] = sum_kt
                        metrics_for_stats["ktd_n_sum"][epoch][query] = sum_kt_n
                        metrics_for_stats["ktd_rel"][epoch][query] = rel_kt
                        metrics_for_stats["ktd_n_rel"][epoch][query] = rel_kt_n
                        metrics_for_stats["wc_sum"][epoch][query] = wc_sum
                        metrics_for_stats["wc_n_sum"][epoch][query] = wc_sum_n
                        metrics_for_stats["wc_diff"][epoch][query] = wc_diff
                        metrics_for_stats["wc_n_diff"][epoch][query] = wc_diff_n
                        metrics_for_stats["wc_rel"][epoch][query] = wc_rel
                        metrics_for_stats["wc_n_rel"][epoch][query] = wc_rel_n
                        metrics_for_stats["wc"][epoch][query] = wc
                        metrics_for_stats["rbo"][epoch][query] = rbo
                    last_list_index_svm[query] = current_list_svm

            averaged_metrics = self.average_metrics_for_queries(metrics_for_stats, list(set(queries)))
            sum_kt_svm = [averaged_metrics["ktd_sum"][q] for q in averaged_metrics["ktd_sum"]]
            sum_kt_svm_n = [averaged_metrics["ktd_n_sum"][q] for q in averaged_metrics["ktd_n_sum"]]
            diff_kt_svm = [averaged_metrics["ktd_diff"][q] for q in averaged_metrics["ktd_diff"]]
            diff_kt_svm_n = [averaged_metrics["ktd_n_diff"][q] for q in averaged_metrics["ktd_n_diff"]]
            rel_kt_svm = [averaged_metrics["ktd_rel"][q] for q in averaged_metrics["ktd_rel"]]
            rel_kt_svm_n = [averaged_metrics["ktd_n_rel"][q] for q in averaged_metrics["ktd_n_rel"]]
            change_rate_sum = [averaged_metrics["wc_sum"][q] for q in averaged_metrics["wc_sum"]]
            change_rate_sum_n = [averaged_metrics["wc_n_sum"][q] for q in averaged_metrics["wc_n_sum"]]
            change_rate_svm_epochs = [averaged_metrics["wc"][q] for q in averaged_metrics["wc"]]
            change_rate_rel = [averaged_metrics["wc_rel"][q] for q in averaged_metrics["wc_rel"]]
            change_rate_rel_n = [averaged_metrics["wc_n_rel"][q] for q in averaged_metrics["wc_n_rel"]]
            change_rate_diff = [averaged_metrics["wc_diff"][q] for q in averaged_metrics["wc_diff"]]
            change_rate_diff_n = [averaged_metrics["wc_n_diff"][q] for q in averaged_metrics["wc_n_diff"]]
            kt_svm = [averaged_metrics["ktd"][q] for q in averaged_metrics["ktd"]]
            rbo_min = [averaged_metrics["rbo"][q] for q in averaged_metrics["rbo"]]
            kendall[svm] = (kt_svm, sum_kt_svm, sum_kt_svm_n, diff_kt_svm, diff_kt_svm_n, rel_kt_svm, rel_kt_svm_n)
            rbo_min_models[svm] = (rbo_min,)
            change_rate[svm] = (
                change_rate_sum, change_rate_sum_n, change_rate_diff, change_rate_diff_n, change_rate_rel,
                change_rate_rel_n, change_rate_svm_epochs)
            t_test = open("svm_averaged_metrics", 'wb')
            pickle.dump(averaged_metrics, t_test)
            t_test.close()
            with open("svm_robustness_stats", "wb") as f:
                pickle.dump(metrics_for_stats, f)

        return kendall, change_rate, rbo_min_models

    def average_metrics_for_queries(self, metrics, queries):
        averaged_epochs = {m: {} for m in metrics}
        for metric in metrics:
            for query in queries:
                res = [metrics[metric][e][query] for e in metrics[metric] if
                       query in list(metrics[metric][e].keys()) and e != 1]
                if not res:
                    continue
                averaged_epochs[metric][query] = np.mean(
                    [metrics[metric][e][query] for e in metrics[metric] if
                     query in list(metrics[metric][e].keys()) and e != 1])
        print(averaged_epochs['wc'])
        return averaged_epochs


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

