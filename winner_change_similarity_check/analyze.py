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
        for svm in scores:
            for epoch in scores[svm]:
                name = "svm" + svm.split("svm_model")[1]
                f = open(name + str(epoch) + ".txt", 'w')
                for query in scores[svm][epoch]:
                    for doc in scores[svm][epoch][query]:
                        f.write(str(query).zfill(3) + " Q0 " + "ROUND-0" + str(epoch) + "-" + str(query).zfill(
                            3) + "-" + doc + " " + str(scores[svm][epoch][query][doc]) + " " + str(
                            scores[svm][epoch][query][doc]) + " seo\n")
                f.close()
                self.order_trec_file(name + str(epoch) + ".txt")

    def cosine_similarity(self, v1, v2):
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i];
            y = v2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        return sumxy / math.sqrt(sumxx * sumyy)

    def calculate_metrics(self, models):
        metrics = {}
        for svm in models:
            ndcg_by_epochs = []
            map_by_epochs = []
            mrr_by_epochs = []
            for i in range(1, 9):
                name = "svm" + svm.split("svm_model")[1]

                score_file = name + str(i)
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
            metrics[svm] = (ndcg_by_epochs, map_by_epochs, mrr_by_epochs)
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

    def create_table(self, competition_data, models, banned_queries):

        scores = self.get_all_scores(models, competition_data)
        rankings, ranks = self.retrieve_ranking(scores)
        bins_for_new_winner_self_similarity, bins_for_winner_similarity, total_self, total_to_winner = self.calculate_average_kendall_tau(
            rankings, ranks, competition_data)
        with open("bins_stats", 'wb') as f:
            pickle.dump((bins_for_new_winner_self_similarity, bins_for_winner_similarity, total_self, total_to_winner),
                        f)

    def bin_creator(self, epochs):
        bins = {i: {} for i in epochs if i != 1}

        for epoch in epochs:
            if epoch == 1:
                continue
            jumps = 0.1
            start = 0
            end = 0.1
            for i in range(10):
                bins[epoch][(start, end)] = 0
                start = round(end, 3)
                end += jumps
                end = round(end, 3)
                end += jumps
        return bins

    def calculate_average_kendall_tau(self, rankings, ranks, competition_data):
        epochs = list(competition_data.keys())
        bins_for_new_winner_self_similarity = self.bin_creator(epochs)
        bins_for_winner_similarity = self.bin_creator(epochs)

        for model in rankings:
            rankings_list_svm = rankings[model]
            last_list_index_svm = {}
            epochs = sorted(list(rankings_list_svm.keys()))
            for epoch in epochs:
                if epoch == 1:
                    continue
                for query in rankings_list_svm[epoch]:
                    current_list_svm = rankings_list_svm[epoch][query]
                    if not last_list_index_svm.get(query, False):
                        last_list_index_svm[query] = current_list_svm
                        continue
                    if ranks[model][epoch][query][0] != ranks[model][epoch - 1][query][0]:
                        new_winner = ranks[model][epoch][query][0]
                        former_winner = ranks[model][epoch - 1][query][0]
                        new_winner_current_vec = competition_data[epoch][query][new_winner]
                        new_winner_former_vec = competition_data[epoch - 1][query][new_winner]
                        former_winner_vec = competition_data[epoch][query][former_winner]
                        self_similarity = self.cosine_similarity(new_winner_current_vec, new_winner_former_vec)
                        similarity_to_winner = self.cosine_similarity(new_winner_current_vec, former_winner_vec)
                        for start, end in bins_for_new_winner_self_similarity[epoch]:
                            if self_similarity >= start and self_similarity <= end:
                                bins_for_new_winner_self_similarity[epoch][(start, end)] += 1
                            if similarity_to_winner >= start and similarity_to_winner <= end:
                                bins_for_winner_similarity[epoch][(start, end)] += 1
            total_self = {i: 0 for i in bins_for_winner_similarity[1]}
            total_to_winner = {i: 0 for i in bins_for_winner_similarity[1]}
            for epoch in bins_for_winner_similarity:
                self_hist_values_sum = sum(list(bins_for_new_winner_self_similarity[epoch].values()))
                winner_to_winner_hist_values_sum = sum(list(bins_for_new_winner_self_similarity[epoch].values()))
                for key in bins_for_new_winner_self_similarity[epoch]:
                    bins_for_new_winner_self_similarity[epoch][key] = float(
                        bins_for_new_winner_self_similarity[epoch][key]) / self_hist_values_sum
                    total_self[key] += bins_for_new_winner_self_similarity[epoch][key]

                    bins_for_winner_similarity[epoch][key] = float(
                        bins_for_winner_similarity[epoch][key]) / winner_to_winner_hist_values_sum
                    total_to_winner[key] += bins_for_winner_similarity[epoch][key]
            for key in total_self:
                total_self[key] = float(total_self[key]) / len(epochs)
                total_to_winner[key] = float(total_to_winner[key]) / len(epochs)
        return bins_for_new_winner_self_similarity, bins_for_winner_similarity, total_self, total_to_winner

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
                rank_vector.append(5 - (sorted_list.index(doc)))
            except:
                print(original_list, sorted_list)
        return rank_vector

    def get_competitors(self, scores_svm):
        competitors = {}
        for query in scores_svm[1]:
            competitors[query] = scores_svm[1][query].keys()
        return competitors

    def retrieve_scores(self, score_file, order, epoch, result):
        with open(score_file, 'r') as scores:
            index = 0
            for score in scores:
                value = float(score.split()[2])
                doc, query = tuple(order[epoch][index].split("@@@"))
                result[epoch][query][doc] = value
                index += 1
        return result

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
