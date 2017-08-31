import svm_ent_models_handler
import svm_models_handler
import numpy as np
from scipy.stats import kendalltau
class analysis:

    def __init__(self):
        ""


    def get_all_scores(self,svm,svm_ent,competition_data):
        scores_svm = {}
        scores_svm_ent = {}
        epochs = range(1,9)
        for epoch in epochs:
            scores_svm[epoch] = {}
            scores_svm_ent[epoch] = {}
            for query in competition_data[epoch]:
                scores_svm[epoch][query] = {}
                scores_svm_ent[epoch][query] = {}
                fold = svm.query_to_fold_index[int(query)]
                weights_svm = svm.weights_index[fold]
                weights_svm_ent = svm_ent.weights_index[fold]
                for doc in competition_data[epoch][query]:
                    features_vector = competition_data[epoch][query][doc]
                    scores_svm[epoch][query][doc] = np.dot(features_vector, weights_svm.T)
                    scores_svm_ent[epoch][query][doc] = np.dot(features_vector, weights_svm_ent.T)
        return scores_svm,scores_svm_ent

    def get_competitors(self,scores_svm):
        competitors={}
        for query in scores_svm[1]:
            competitors[query] = scores_svm[1][query].keys()
        return competitors


    def retrieve_ranking(self,scores_svm, scores_svm_ent):
        rankings_svm = {}
        rankings_svm_ent = {}
        competitors = self.get_competitors(scores_svm)
        for epoch in scores_svm:
            rankings_svm[epoch]={}
            rankings_svm_ent[epoch]={}
            for query in scores_svm[epoch]:
                retrieved_list_svm = sorted(competitors[query],key=lambda x:scores_svm[epoch][query][x],reverse=True)
                rankings_svm[epoch][query]= self.transition_to_rank_vector(competitors[query],retrieved_list_svm)
                retrieved_list_svm_ent =  sorted(competitors[query],key=lambda x:scores_svm_ent[epoch][query][x],reverse=True)
                rankings_svm_ent[epoch][query] =self.transition_to_rank_vector(competitors[query],retrieved_list_svm_ent)
        return retrieved_list_svm_ent,retrieved_list_svm

    def transition_to_rank_vector(self,original_list,sorted_list):
        rank_vector = []
        for doc in original_list:
            rank_vector.append(sorted_list.index(doc) + 1)
        return rank_vector


    def calculate_average_kendall_tau(self,retrieved_list_svm_ent, retrieved_list_svm):
        kt_svm = []
        kt_svm_ent=[]
        kt_svm_orig = []
        kt_svm_ent_orig = []
        last_list_index_svm={}

        last_list_index_svm_ent = {}
        original_list_index_svm = {}
        original_list_index_svm_ent = {}
        n_q = len(retrieved_list_svm_ent[1])

        for epoch in retrieved_list_svm:
            sum_svm = 0
            sum_svm_ent =0
            sum_svm_original = 0
            sum_svm_ent_original = 0
            for query in retrieved_list_svm[epoch]:
                current_list_svm = retrieved_list_svm[epoch][query]
                current_list_svm_ent = retrieved_list_svm_ent[epoch][query]
                if not last_list_index_svm.get(query,False):
                    last_list_index_svm[query]=current_list_svm
                    last_list_index_svm_ent[query]=current_list_svm_ent
                    original_list_index_svm[query]=current_list_svm_ent
                    original_list_index_svm_ent[query]=current_list_svm_ent
                    continue
                kt = kendalltau(original_list_index_svm[query], current_list_svm)
                kt_orig = kendalltau(last_list_index_svm[query], current_list_svm)
                if not np.isnan(kt):
                    sum_svm+=kt
                if not np.isnan(kt_orig):
                    sum_svm_original+=kt_orig
                kt_ent = kendalltau(last_list_index_svm_ent[query],current_list_svm_ent)
                kt_ent_orig = kendalltau(original_list_index_svm_ent[query],current_list_svm_ent)
                if not np.isnan(kt_ent_orig):
                    sum_svm_ent_original += kt_ent_orig
                if not np.isnan(kt_ent):
                    sum_svm_ent+=kt_ent
            kt_svm.append(float(sum_svm)/n_q)
            kt_svm_orig.append(float(sum_svm_original)/n_q)
            kt_svm_ent.append(float(sum_svm_ent)/n_q)
            kt_svm_ent_orig.append(float(sum_svm_ent_original)/n_q)
        return kt_svm,kt_svm_ent,kt_svm_orig,kt_svm_ent_orig


    def analyze(self,svm,svm_ent,competition_data):
        scores_svm, scores_svm_ent = self.get_all_scores(svm,svm_ent,competition_data)
        retrieved_list_svm_ent, retrieved_list_svm = self.retrieve_ranking(scores_svm, scores_svm_ent)





