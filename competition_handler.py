import models_handler as mh
import params
import optimizer as o
import numpy as np
from scipy.stats import kendalltau as kt


class competition_handler:
    def __init__(self,alpha):
        self.alpha = alpha
        self.optimizer_index = {}

    def create_optimizer_index(self,model_handler):
        for i in range(params.number_of_folds):
            fold = i+1
            self.optimizer_index[fold] = o.optimizer(self.alpha,model_handler.weights_index[fold])



    def update_competitors(self,competitors,model_handler,X):
        competitors_new = {}
        for query in competitors:
            competitors_new[query]={}
            docs = competitors[query]
            for doc in docs:
                fold = model_handler.query_to_fold_index[query]
                new_doc = self.optimizer_index[fold].get_best_features(X[doc])
                competitors_new[query][doc]=new_doc
        return competitors_new

    def update_rankings(self,new_competitors,model_handler,competitors):
        rankings = {}
        for query in new_competitors:
            doc_scores = {}
            weights = model_handler.weights_index[model_handler.query_to_fold_index[query]]
            original_rank = competitors[query]
            for doc in new_competitors[query]:
                doc_features = new_competitors[query][doc]
                score = np.dot(doc_features, weights.T)
                doc_scores[doc] = score
            sorted_ranking = sorted(original_rank, key=lambda x: (doc_scores[x]), reverse=True)
            rankings[query]=sorted_ranking
        return rankings


    def retrieve_competitors(self):
        competitors = {}
        queries_finished = {}
        with open(params.score_file) as scores_data:
            for score_record in scores_data:
                data = score_record.split()
                query_number = int(data[0])
                document = data[2]
                if not queries_finished.get(query_number, False):
                    if not competitors.get(query_number, False):
                        competitors[query_number] = []
                    if len(competitors[query_number]) < params.number_of_competitors:
                        competitors[query_number].append(int(document))
                    else:
                        queries_finished[query_number] = True
        return competitors


    def get_kendall_tau_measures(self,rankings,competitors):
        original_rank = range(1,params.number_of_competitors+1)
        kendall_tau = []
        for query in rankings:
            ranked_list = rankings[query]
            rank_vector = self.transition_to_rank_vector(query,competitors[query],ranked_list)
            kendall, p_value = kt(original_rank, rank_vector)
            kendall_tau.append(kendall)
        mean = np.mean(kendall_tau)
        std = np.std(kendall_tau)
        return mean,std



    def transition_to_rank_vector(self, query, reference_of_indexes, ranked_list):
        doc_list = reference_of_indexes[query]
        rank_vector = []
        for doc in doc_list:
            rank_vector.append(ranked_list.index(doc) + 1)
        return rank_vector


    def competition(self,model_handler,X):
        self.create_optimizer_index(model_handler)
        competitors = self.retrieve_competitors()
        competitors_new = self.update_competitors(competitors,model_handler,X)
        rankings = self.update_rankings(competitors_new,model_handler,competitors)
        kendall_mean,kendall_std = self.get_kendall_tau_measures(rankings,competitors)








