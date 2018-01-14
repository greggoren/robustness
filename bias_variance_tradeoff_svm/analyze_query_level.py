import pickle
import itertools
import subprocess
import numpy as np
from scipy.stats import kendalltau
import RBO as r
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from kendall_tau import weighted_kendall_tau


def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, shell=True
                         )
    return iter(p.stdout.readline, '')


def run_bash_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, shell=True)
    out, err = p.communicate()
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
            queries = []
            index = 0
            for query in competition_data[epoch]:
                for doc in competition_data[epoch][query]:
                    data_set.append(competition_data[epoch][query][doc])
                    queries.append(query)
                    order[epoch][index] = doc + "@@@" + query
                    index += 1
            features_file = "features" + str(epoch)
            self.create_data_set_file(data_set, queries, features_file)
            for model in models:
                score_file = self.run_lambda_mart(features_file, epoch, model)
                scores[model] = self.retrieve_scores(score_file, order, epoch, scores[model])
        return scores

    # s
    def order_trec_file(self, trec_file):
        final = trec_file.replace(".txt", "")
        command = "sort -k1,1 -k5nr -k2,1 " + trec_file + " > " + final
        for line in run_bash_command(command):
            print(line)
        command = "rm " + trec_file
        for line in run_bash_command(command):
            print(line)
        return final

    def extract_score(self, scores):
        for model in scores:
            for epoch in scores[model]:
                name = model.split("svm_model")[1]

                f = open(name + "_" + str(epoch) + ".txt", 'w')
                for query in scores[model][epoch]:
                    for doc in scores[model][epoch][query]:
                        f.write(str(query).zfill(3) + " Q0 " + "ROUND-0" + str(epoch) + "-" + str(query).zfill(
                            3) + "-" + doc + " " + str(scores[model][epoch][query][doc]) + " " + str(
                            scores[model][epoch][query][doc]) + " seo\n")
                f.close()
                self.order_trec_file(name + "_" + str(epoch) + ".txt")

    def calculate_metrics(self, models):
        metrics = {}
        for model in models:
            ndcg_by_epochs = []
            map_by_epochs = []
            mrr_by_epochs = []
            for i in range(1, 9):
                name = model.split("model_")[1]

                score_file = name + "_" + str(i)
                qrels = "../rel3/rel0" + str(i)

                command = "../trec_eval -m ndcg " + qrels + " " + score_file
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
        scores = self.get_all_scores(models, competition_data)
        rankings, ranks = self.retrieve_ranking(scores)
        kendall, change_rate, rbo_min_models = self.calculate_average_kendall_tau(rankings, banned_queries, weights,
                                                                                  ranks)
        # self.extract_score(scores)
        kendall_for_pearson = {i: {} for i in ["reg", "max", "mean"]}
        C_for_pearson = {}
        # metrics = self.calculate_metrics(scores)
        rbo_for_pearson = {}
        wc_for_pearson = {"reg": {}, "winner": {}}
        final_correlation_spearman = {j: {} for j in ["C"]}
        final_correlation_pearson = {j: {} for j in ["C"]}
        query_correlation_pearson = {
            j: {i: {} for i in ["kendall_max", "kendall_mean", "kendall_reg", "wc_reg", "wc_winner", "rbo"]} for j in
            ["C"]}
        query_correlation_spearman = {
            j: {i: {} for i in ["kendall_max", "kendall_mean", "kendall_reg", "wc_reg", "wc_winner", "rbo"]} for j in
            ["C"]}
        for query in kendall["reg"]:
            kendall_for_pearson["reg"][query] = []
            kendall_for_pearson["max"][query] = []
            kendall_for_pearson["mean"][query] = []
            C_for_pearson[query] = []
            rbo_for_pearson[query] = []
            wc_for_pearson["reg"][query] = []
            wc_for_pearson["winner"][query] = []
            for model in kendall["reg"][query]:
                kendall_for_pearson["reg"][query].append(kendall["reg"][query][model])
                kendall_for_pearson["max"][query].append(kendall["max"][query][model])
                kendall_for_pearson["mean"][query].append(kendall["mean"][query][model])
                C = model.split("svm_model")[1]
                C_for_pearson[query].append(float(C))
                wc_for_pearson["reg"][query].append(change_rate[query][model]["reg"])
                wc_for_pearson["winner"][query].append(change_rate[query][model]["winner"])
                rbo_for_pearson[query].append(rbo_min_models[query][model])
            # print(wc_for_pearson["reg"]["182"])
            query_correlation_pearson["C"]["kendall_reg"][query] = pearsonr(C_for_pearson[query],
                                                                            kendall_for_pearson["reg"][query])
            query_correlation_pearson["C"]["kendall_max"][query] = pearsonr(C_for_pearson[query],
                                                                            kendall_for_pearson["max"][query])
            query_correlation_pearson["C"]["kendall_mean"][query] = pearsonr(C_for_pearson[query],
                                                                             kendall_for_pearson["mean"][query])
            query_correlation_pearson["C"]["wc_reg"][query] = pearsonr(C_for_pearson[query],
                                                                       wc_for_pearson["reg"][query])
            query_correlation_pearson["C"]["wc_winner"][query] = pearsonr(C_for_pearson[query],
                                                                          wc_for_pearson["winner"][query])
            query_correlation_pearson["C"]["rbo"][query] = pearsonr(C_for_pearson[query],
                                                                    rbo_for_pearson[query])
            query_correlation_spearman["C"]["kendall_reg"][query] = spearmanr(C_for_pearson[query],
                                                                              kendall_for_pearson["reg"][query])
            query_correlation_spearman["C"]["kendall_max"][query] = spearmanr(C_for_pearson[query],
                                                                              kendall_for_pearson["max"][query])
            query_correlation_spearman["C"]["kendall_mean"][query] = spearmanr(C_for_pearson[query],
                                                                               kendall_for_pearson["mean"][query])
            query_correlation_spearman["C"]["wc_reg"][query] = spearmanr(C_for_pearson[query],
                                                                         wc_for_pearson["reg"][query])
            query_correlation_spearman["C"]["wc_winner"][query] = spearmanr(C_for_pearson[query],
                                                                            wc_for_pearson["winner"][query])
            query_correlation_spearman["C"]["rbo"][query] = spearmanr(C_for_pearson[query],
                                                                      rbo_for_pearson[query])

        final_correlation_pearson["C"]["kendall_reg"] = (
            np.mean([query_correlation_pearson["C"]["kendall_reg"][q][0] for q in
                     query_correlation_pearson["C"]["kendall_reg"]]),
            np.mean([query_correlation_pearson["C"]["kendall_reg"][q][1] for q in
                     query_correlation_pearson["C"]["kendall_reg"]]))
        final_correlation_pearson["C"]["kendall_max"] = (
            np.mean([query_correlation_pearson["C"]["kendall_max"][q][0] for q in
                     query_correlation_pearson["C"]["kendall_max"]]),
            np.mean([query_correlation_pearson["C"]["kendall_max"][q][1] for q in
                     query_correlation_pearson["C"]["kendall_max"]]))
        final_correlation_pearson["C"]["kendall_mean"] = (
            np.mean([query_correlation_pearson["C"]["kendall_mean"][q][0] for q in
                     query_correlation_pearson["C"]["kendall_mean"]]),
            np.mean([query_correlation_pearson["C"]["kendall_mean"][q][1] for q in
                     query_correlation_pearson["C"]["kendall_mean"]]))
        final_correlation_pearson["C"]["wc_reg"] = (
            np.mean([query_correlation_pearson["C"]["wc_reg"][q][0] for q in
                     query_correlation_pearson["C"]["wc_reg"]]),
            np.mean([query_correlation_pearson["C"]["wc_reg"][q][1] for q in
                     query_correlation_pearson["C"]["wc_reg"]]))
        final_correlation_pearson["C"]["wc_winner"] = (
            np.mean([query_correlation_pearson["C"]["wc_winner"][q][0] for q in
                     query_correlation_pearson["C"]["wc_winner"]]),
            np.mean([query_correlation_pearson["C"]["wc_winner"][q][1] for q in
                     query_correlation_pearson["C"]["wc_winner"]]))
        final_correlation_pearson["C"]["rbo"] = (
            np.mean(
                [query_correlation_pearson["C"]["rbo"][q][0] for q in query_correlation_pearson["C"]["rbo"]]),
            np.mean([query_correlation_pearson["C"]["rbo"][q][1] for q in
                     query_correlation_pearson["C"]["rbo"]]))
        final_correlation_spearman["C"]["kendall_reg"] = (
            np.mean([query_correlation_spearman["C"]["kendall_reg"][q][0] for q in
                     query_correlation_spearman["C"]["kendall_reg"]]),
            np.mean([query_correlation_spearman["C"]["kendall_reg"][q][1] for q in
                     query_correlation_spearman["C"]["kendall_reg"]]))
        final_correlation_spearman["C"]["kendall_mean"] = (
            np.mean([query_correlation_spearman["C"]["kendall_mean"][q][0] for q in
                     query_correlation_spearman["C"]["kendall_mean"]]),
            np.mean([query_correlation_spearman["C"]["kendall_mean"][q][1] for q in
                     query_correlation_spearman["C"]["kendall_mean"]]))
        final_correlation_spearman["C"]["kendall_max"] = (
            np.mean([query_correlation_spearman["C"]["kendall_max"][q][0] for q in
                     query_correlation_spearman["C"]["kendall_max"]]),
            np.mean([query_correlation_spearman["C"]["kendall_max"][q][1] for q in
                     query_correlation_spearman["C"]["kendall_max"]]))
        final_correlation_spearman["C"]["wc_reg"] = (
            np.mean(
                [query_correlation_spearman["C"]["wc_reg"][q][0] for q in
                 query_correlation_spearman["C"]["wc_reg"]]),
            np.mean([query_correlation_spearman["C"]["wc_reg"][q][1] for q in
                     query_correlation_spearman["C"]["wc_reg"]]))
        final_correlation_spearman["C"]["wc_winner"] = (
            np.mean(
                [query_correlation_spearman["C"]["wc_winner"][q][0] for q in
                 query_correlation_spearman["C"]["wc_winner"]]),
            np.mean([query_correlation_spearman["C"]["wc_winner"][q][1] for q in
                     query_correlation_spearman["C"]["wc_winner"]]))
        final_correlation_spearman["C"]["rbo"] = (
            np.mean(
                [query_correlation_spearman["C"]["rbo"][q][0] for q in query_correlation_spearman["C"]["rbo"]]),
            np.mean([query_correlation_spearman["C"]["rbo"][q][1] for q in
                     query_correlation_spearman["C"]["rbo"]]))
        print(query_correlation_spearman["C"]["wc_reg"])
        f = open("pearson_C.tex", 'w')
        f.write("\\begin{tabular}{c|c|c}\n")
        f.write("Metric & Correlation & P-value \\\\ \n")
        corr = final_correlation_pearson["C"]["kendall_reg"]
        f.write("Kendall-$\\tau$ & " + str(corr[0]) + " & " + str(corr[1]) + " \\\\ \n")
        corr = final_correlation_pearson["C"]["kendall_max"]
        f.write("Kendall-$\\tau$ max & " + str(corr[0]) + " & " + str(corr[1]) + " \\\\ \n")
        corr = final_correlation_pearson["C"]["kendall_mean"]
        f.write("Kendall-$\\tau$ mean & " + str(corr[0]) + " & " + str(corr[1]) + " \\\\ \n")
        corr = final_correlation_pearson["C"]["wc_reg"]
        f.write("Winner change & " + str(corr[0]) + " & " + str(corr[1]) + " \\\\ \n")
        corr = final_correlation_pearson["C"]["wc_winner"]
        f.write("Winner change new winner norm & " + str(corr[0]) + " & " + str(corr[1]) + " \\\\ \n")
        corr = final_correlation_pearson["C"]["rbo"]
        f.write("RBO & " + str(corr[0]) + " & " + str(corr[1]) + " \\\\ \n")
        f.write("\\end{tabular}")
        f = open("spearman_C.tex", 'w')
        f.write("\\begin{tabular}{c|c|c}\n")
        f.write("Metric & Correlation & P-value \\\\ \n")
        corr = final_correlation_spearman["C"]["kendall_reg"]
        f.write("Kendall-$\\tau$ & " + str(corr[0]) + " & " + str(corr[1]) + " \\\\ \n")
        corr = final_correlation_spearman["C"]["kendall_max"]
        f.write("Kendall-$\\tau$ max & " + str(corr[0]) + " & " + str(corr[1]) + " \\\\ \n")
        corr = final_correlation_spearman["C"]["kendall_mean"]
        f.write("Kendall-$\\tau$ mean & " + str(corr[0]) + " & " + str(corr[1]) + " \\\\ \n")
        corr = final_correlation_spearman["C"]["wc_reg"]
        f.write("Winner change & " + str(corr[0]) + " & " + str(corr[1]) + " \\\\ \n")
        corr = final_correlation_spearman["C"]["wc_winner"]
        f.write("Winner change new winner norm & " + str(corr[0]) + " & " + str(corr[1]) + " \\\\ \n")
        corr = final_correlation_spearman["C"]["rbo"]
        f.write("RBO & " + str(corr[0]) + " & " + str(corr[1]) + " \\\\ \n")
        f.write("\\end{tabular}")

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
        kendall = {i: {} for i in ["reg", "max", "mean"]}
        change_rate = {}
        rbo_min_models = {}
        for model in rankings:
            rankings_list_lm = rankings[model]
            last_list_index_lm = {}

            epochs = sorted(list(rankings_list_lm.keys()))
            for epoch in epochs:
                for query in rankings_list_lm[epoch]:
                    if not kendall["reg"].get(query, False):
                        kendall["reg"][query] = {}
                        kendall["max"][query] = {}
                        kendall["mean"][query] = {}
                        change_rate[query] = {}
                        rbo_min_models[query] = {}
                    if not kendall["reg"][query].get(model, False):
                        kendall["reg"][query][model] = []
                        kendall["mean"][query][model] = []
                        kendall["max"][query][model] = []
                        change_rate[query][model] = {"reg": [], "winner": []}
                        rbo_min_models[query][model] = []
                    current_list_svm = rankings_list_lm[epoch][query]
                    if not last_list_index_lm.get(query, False):
                        last_list_index_lm[query] = current_list_svm
                        continue
                    if ranks[model][epoch][query][0] != ranks[model][epoch - 1][query][0]:
                        change_rate[query][model]["reg"].append(1)
                        change_rate[query][model]["winner"].append(
                            float(1) / (weights[epoch][query][ranks[model][epoch][query][0]] + 1))
                    else:
                        change_rate[query][model]["reg"].append(0)
                        change_rate[query][model]["winner"].append(0)
                    kt = kendalltau(current_list_svm, last_list_index_lm[query])[0]
                    kt_max = weighted_kendall_tau(ranks[model][epoch][query], ranks[model][epoch - 1][query],
                                                  weights[epoch][query],
                                                  "max")
                    kt_mean = weighted_kendall_tau(ranks[model][epoch][query], ranks[model][epoch - 1][query],
                                                   weights[epoch][query],
                                                   "mean")
                    if not np.isnan(kt):
                        kendall["reg"][query][model].append(kt)
                    else:
                        print("query", query)
                    kendall["max"][query][model].append(kt_max)
                    kendall["mean"][query][model].append(kt_mean)
                    rbo = r.rbo_dict({x: j for x, j in enumerate(last_list_index_lm[query])},
                                     {x: j for x, j in enumerate(current_list_svm)}, 0.7)["min"]
                    rbo_min_models[query][model].append(rbo)
                    last_list_index_lm[query] = current_list_svm
        for query in kendall["reg"]:
            for model in kendall["reg"][query]:
                kendall["reg"][query][model] = np.mean(kendall["reg"][query][model])
                kendall["max"][query][model] = np.mean(kendall["max"][query][model])
                kendall["mean"][query][model] = np.mean(kendall["mean"][query][model])
                rbo_min_models[query][model] = np.mean(rbo_min_models[query][model])
                change_rate[query][model]["reg"] = np.mean(change_rate[query][model]["reg"])
                change_rate[query][model]["winner"] = np.mean(change_rate[query][model]["winner"])
        return kendall, change_rate, rbo_min_models

    def get_all_scores(self, svms, competition_data):
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

    def transition_to_rank_vector(self, original_list, sorted_list):
        rank_vector = []
        for doc in original_list:
            try:
                rank_vector.append(6 - (sorted_list.index(doc) + 1))
                # rank_vector.append(sorted_list.index(doc) + 1)
            except:
                print(original_list, sorted_list)
        return rank_vector

    def get_competitors(self, scores_svm):
        competitors = {}
        for query in scores_svm[1]:
            competitors[query] = scores_svm[1][query].keys()
        return competitors

    def create_data_set_file(self, X, queries, feature_file_name):
        with open(feature_file_name, 'w') as feature_file:
            for i, doc in enumerate(X):
                features = " ".join([str(a + 1) + ":" + str(b) for a, b in enumerate(doc)])
                line = "1 qid:" + queries[i] + " " + features + "\n"
                feature_file.write(line)

    def retrieve_scores(self, score_file, order, epoch, result):
        with open(score_file, 'r') as scores:
            index = 0
            for score in scores:
                value = float(score.split()[2])
                doc, query = tuple(order[epoch][index].split("@@@"))
                result[epoch][query][doc] = value
                index += 1
        return result

    def run_lambda_mart(self, features, epoch, model):
        java_path = "/lv_local/home/sgregory/jdk1.8.0_121/bin/java"
        jar_path = "/lv_local/home/sgregory/SEO_CODE/model_running/RankLib.jar"
        score_file = "score" + str(epoch)
        features = features
        model_path = model
        command = java_path + " -jar " + jar_path + " -load " + model_path + " -rank " + features + " -score " + score_file
        run_bash_command(command)
        return score_file

    def retrive_qrel(self, qrel_file):
        qrel = {}
        with open(qrel_file) as qrels:
            for q in qrels:
                splited = q.split()
                name = splited[2]
                epoch = int(name.split("-")[1])
                query = name.split("-")[2]
                doc = name.split("-")[3]
                rel = splited[3].rstrip()
                if not qrel.get(epoch, False):
                    qrel[epoch] = {}
                if not qrel[epoch].get(query, False):
                    qrel[epoch][query] = {}
                qrel[epoch][query][doc] = rel
        return qrel

    def mrr(self, qrel, rankings):
        mrr_for_ranker = {}
        for ranker in rankings:
            mrr_by_epochs = []
            for epoch in rankings[ranker]:
                mrr = 0
                nq = 0
                for query in rankings[ranker][epoch]:
                    if query.__contains__("_2"):
                        continue
                    nq += 1
                    ranking_list = rankings[ranker][epoch][query]
                    try:
                        for doc in ranking_list:
                            if qrel[epoch][query][doc] != "0":
                                mrr += (1.0 / (ranking_list.index(doc) + 1))
                                break
                    except:

                        print(qrel.keys())
                        print(epoch)
                        print(query)
                        print(doc)
                mrr_by_epochs.append(mrr / nq)
            mrr_for_ranker[ranker] = mrr_by_epochs
        return mrr_for_ranker
