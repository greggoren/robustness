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
                name = model.split("model_")[1]

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
            for i in range(1, 6):
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
        scores = self.create_lambdaMart_scores(competition_data, models)
        rankings, ranks = self.retrieve_ranking(scores)
        kendall, change_rate, rbo_min_models = self.calculate_average_kendall_tau(rankings, banned_queries, weights,
                                                                                  ranks)
        self.extract_score(scores)
        kendall_for_pearson = {}
        trees_for_pearson = {}
        leaves_for_pearson = {}
        # metrics = self.calculate_metrics(scores)
        rbo_for_pearson = {}
        wc_for_pearson = {}
        final_correlation_spearman = {j: {} for j in ["trees", "leaves"]}
        final_correlation_pearson = {j: {} for j in ["trees", "leaves"]}
        query_correlation_pearson = {j: {i: {} for i in ["kendall", "wc", "rbo"]} for j in ["trees", "leaves"]}
        query_correlation_spearman = {j: {i: {} for i in ["kendall", "wc", "rbo"]} for j in ["trees", "leaves"]}
        for query in kendall:
            kendall_for_pearson[query] = []
            trees_for_pearson[query] = []
            leaves_for_pearson[query] = []
            rbo_for_pearson[query] = []
            wc_for_pearson[query] = []
            for model in kendall[query]:
                kendall_for_pearson[query].append(kendall[query][model])
                trees, leaves = tuple((model.split("model_")[1].split("_")[0], model.split("model_")[1].split("_")[1]))
                trees_for_pearson[query].append(int(trees))
                leaves_for_pearson[query].append(int(leaves))
                wc_for_pearson[query].append(change_rate[query][model])
                rbo_for_pearson[query].append(rbo_min_models[query][model])

            query_correlation_pearson["trees"]["kendall"][query] = pearsonr(trees_for_pearson[query],
                                                                            kendall_for_pearson[query])
            query_correlation_pearson["trees"]["wc"][query] = pearsonr(trees_for_pearson[query], wc_for_pearson[query])
            query_correlation_pearson["trees"]["rbo"][query] = pearsonr(trees_for_pearson[query],
                                                                        rbo_for_pearson[query])
            query_correlation_spearman["trees"]["kendall"][query] = spearmanr(trees_for_pearson[query],
                                                                     kendall_for_pearson[query])
            query_correlation_spearman["trees"]["wc"][query] = spearmanr(trees_for_pearson[query],
                                                                         wc_for_pearson[query])
            query_correlation_spearman["trees"]["rbo"][query] = spearmanr(trees_for_pearson[query],
                                                                          rbo_for_pearson[query])
            query_correlation_pearson["leaves"]["kendall"][query] = pearsonr(leaves_for_pearson[query],
                                                                             kendall_for_pearson[query])
            query_correlation_pearson["leaves"]["wc"][query] = pearsonr(leaves_for_pearson[query],
                                                                        wc_for_pearson[query])
            query_correlation_pearson["leaves"]["rbo"][query] = pearsonr(leaves_for_pearson[query],
                                                                         rbo_for_pearson[query])
            query_correlation_spearman["leaves"]["kendall"][query] = spearmanr(leaves_for_pearson[query],
                                                                               kendall_for_pearson[query])
            query_correlation_spearman["leaves"]["wc"][query] = spearmanr(leaves_for_pearson[query],
                                                                          wc_for_pearson[query])
            query_correlation_spearman["leaves"]["rbo"][query] = spearmanr(leaves_for_pearson[query],
                                                                           rbo_for_pearson[query])
        final_correlation_pearson["trees"]["kendall"] = (
            np.mean([query_correlation_pearson["trees"]["kendall"][q][0] for q in
                     query_correlation_pearson["trees"]["kendall"]]),
            np.mean([query_correlation_pearson["trees"]["kendall"][q][1] for q in
                     query_correlation_pearson["trees"]["kendall"]]))
        final_correlation_pearson["trees"]["wc"] = (
            np.mean([query_correlation_pearson["trees"]["wc"][q][0] for q in query_correlation_pearson["trees"]["wc"]]),
            np.mean([query_correlation_pearson["trees"]["wc"][q][1] for q in
                     query_correlation_pearson["trees"]["wc"]]))
        final_correlation_pearson["trees"]["rbo"] = (
            np.mean(
                [query_correlation_pearson["trees"]["rbo"][q][0] for q in query_correlation_pearson["trees"]["rbo"]]),
            np.mean([query_correlation_pearson["trees"]["rbo"][q][1] for q in
                     query_correlation_pearson["trees"]["rbo"]]))
        final_correlation_spearman["trees"]["kendall"] = (
            np.mean([query_correlation_spearman["trees"]["kendall"][q][0] for q in
                     query_correlation_spearman["trees"]["kendall"]]),
            np.mean([query_correlation_spearman["trees"]["kendall"][q][1] for q in
                     query_correlation_spearman["trees"]["kendall"]]))
        final_correlation_spearman["trees"]["wc"] = (
            np.mean(
                [query_correlation_spearman["trees"]["wc"][q][0] for q in query_correlation_spearman["trees"]["wc"]]),
            np.mean([query_correlation_spearman["trees"]["wc"][q][1] for q in
                     query_correlation_spearman["trees"]["wc"]]))
        final_correlation_spearman["trees"]["rbo"] = (
            np.mean(
                [query_correlation_spearman["trees"]["rbo"][q][0] for q in query_correlation_spearman["trees"]["rbo"]]),
            np.mean([query_correlation_spearman["trees"]["rbo"][q][1] for q in
                     query_correlation_spearman["trees"]["rbo"]]))
        final_correlation_pearson["leaves"]["kendall"] = (
            np.mean([query_correlation_pearson["leaves"]["kendall"][q][0] for q in
                     query_correlation_pearson["leaves"]["kendall"]]),
            np.mean([query_correlation_pearson["leaves"]["kendall"][q][1] for q in
                     query_correlation_pearson["leaves"]["kendall"]]))
        final_correlation_pearson["leaves"]["wc"] = (
            np.mean(
                [query_correlation_pearson["leaves"]["wc"][q][0] for q in query_correlation_pearson["leaves"]["wc"]]),
            np.mean(
                [query_correlation_pearson["leaves"]["wc"][q][1] for q in query_correlation_pearson["leaves"]["wc"]]))
        final_correlation_pearson["leaves"]["rbo"] = (
            np.mean(
                [query_correlation_pearson["leaves"]["rbo"][q][0] for q in query_correlation_pearson["leaves"]["rbo"]]),
            np.mean([query_correlation_pearson["leaves"]["rbo"][q][1] for q in
                     query_correlation_pearson["leaves"]["rbo"]]))
        final_correlation_spearman["leaves"]["kendall"] = (
            np.mean([query_correlation_spearman["leaves"]["kendall"][q][0] for q in
                     query_correlation_spearman["leaves"]["kendall"]]),
            np.mean([query_correlation_spearman["leaves"]["kendall"][q][1] for q in
                     query_correlation_spearman["leaves"]["kendall"]]))
        final_correlation_spearman["leaves"]["wc"] = (
            np.mean(
                [query_correlation_spearman["leaves"]["wc"][q][0] for q in query_correlation_spearman["leaves"]["wc"]]),
            np.mean([query_correlation_spearman["leaves"]["wc"][q][1] for q in
                     query_correlation_spearman["leaves"]["wc"]]))
        final_correlation_spearman["leaves"]["rbo"] = (
            np.mean([query_correlation_spearman["leaves"]["rbo"][q][0] for q in
                     query_correlation_spearman["leaves"]["rbo"]]),
            np.mean([query_correlation_spearman["leaves"]["rbo"][q][1] for q in
                     query_correlation_spearman["leaves"]["rbo"]]))

        print(final_correlation_pearson["trees"])
        print(final_correlation_spearman["trees"])
        print(final_correlation_pearson["leaves"])
        print(final_correlation_spearman["leaves"])

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
        for model in rankings:
            rankings_list_lm = rankings[model]
            last_list_index_lm = {}

            epochs = sorted(list(rankings_list_lm.keys()))
            for epoch in epochs:
                for query in rankings_list_lm[epoch]:
                    if not kendall.get(query, False):
                        kendall[query] = {}
                        change_rate[query] = {}
                        rbo_min_models[query] = {}
                    if not kendall[query].get(model, False):
                        kendall[query][model] = []
                        change_rate[query][model] = []
                        rbo_min_models[query][model] = []
                    current_list_svm = rankings_list_lm[epoch][query]
                    if not last_list_index_lm.get(query, False):
                        last_list_index_lm[query] = current_list_svm
                        continue
                    if current_list_svm.index(5) != last_list_index_lm[query].index(5):
                        change_rate[query][model].append(1)
                    else:
                        change_rate[query][model].append(0)
                    kt = kendalltau(current_list_svm, last_list_index_lm[query])[0]
                    if not np.isnan(kt):
                        kendall[query][model].append(kt)
                    rbo = r.rbo_dict({x: j for x, j in enumerate(last_list_index_lm[query])},
                                     {x: j for x, j in enumerate(current_list_svm)}, 0.7)["min"]
                    rbo_min_models[query][model].append(rbo)
                    last_list_index_lm[query] = current_list_svm
        for query in kendall:
            for model in kendall[query]:
                kendall[query][model] = np.mean(kendall[query][model])
                rbo_min_models[query][model] = np.mean(rbo_min_models[query][model])
                change_rate[query][model] = np.mean(change_rate[query][model])
        return kendall, change_rate, rbo_min_models

    def get_all_scores(self, svms, competition_data):
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
                        scores[svm][epoch][query][doc] = np.dot(weights_svm, features_vector.T)
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
