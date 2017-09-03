import svm_ent_models_handler
import svm_models_handler
import numpy as np
from scipy.stats import kendalltau
from scipy import spatial
import itertools
class analysis:

    def __init__(self):
        ""

    def cosine_distance(self,x,y):
        return spatial.distance.cosine(x,y)[0]

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
        return rankings_svm_ent,rankings_svm

    def transition_to_rank_vector(self,original_list,sorted_list):
        rank_vector = []
        for doc in original_list:
            rank_vector.append(sorted_list.index(doc) + 1)
        return rank_vector


    def calculate_average_kendall_tau(self, rankings_svm_ent, rankings_list_svm):
        kt_svm = []
        kt_svm_ent=[]
        kt_svm_orig = []
        kt_svm_ent_orig = []
        last_list_index_svm={}

        last_list_index_svm_ent = {}
        original_list_index_svm = {}
        original_list_index_svm_ent = {}
        change_rate_svm_epochs =[]
        change_rate_svm_ent_epochs =[]
        for epoch in rankings_list_svm:
            sum_svm = 0
            sum_svm_ent =0
            sum_svm_original = 0
            sum_svm_ent_original = 0
            n_q=0
            change_rate_svm = 0
            change_rate_svm_ent = 0
            for query in rankings_list_svm[epoch]:
                current_list_svm = rankings_list_svm[epoch][query]
                current_list_svm_ent = rankings_svm_ent[epoch][query]
                if not last_list_index_svm.get(query,False):
                    last_list_index_svm[query]=current_list_svm
                    last_list_index_svm_ent[query]=current_list_svm_ent
                    original_list_index_svm[query]=current_list_svm
                    original_list_index_svm_ent[query]=current_list_svm_ent
                    continue
                if current_list_svm.index(1)!=last_list_index_svm[query].index(1):
                    change_rate_svm +=1
                if current_list_svm_ent.index(1)!=last_list_index_svm_ent[query].index(1):
                    change_rate_svm_ent +=1
                n_q+=1
                kt = kendalltau(last_list_index_svm[query], current_list_svm)[0]
                kt_orig = kendalltau(original_list_index_svm[query], current_list_svm)[0]
                if not np.isnan(kt):
                    sum_svm+=kt
                if not np.isnan(kt_orig):
                    sum_svm_original+=kt_orig
                kt_ent = kendalltau(last_list_index_svm_ent[query],current_list_svm_ent)[0]
                kt_ent_orig = kendalltau(original_list_index_svm_ent[query],current_list_svm_ent)[0]
                if not np.isnan(kt_ent_orig):
                    sum_svm_ent_original += kt_ent_orig
                if not np.isnan(kt_ent):
                    sum_svm_ent+=kt_ent
                last_list_index_svm[query] = current_list_svm
                last_list_index_svm_ent[query] = current_list_svm_ent

            if n_q==0:
                continue
            change_rate_svm_ent_epochs.append(change_rate_svm_ent)
            change_rate_svm_epochs.append(change_rate_svm)
            kt_svm.append(float(sum_svm)/n_q)
            kt_svm_orig.append(float(sum_svm_original)/n_q)
            kt_svm_ent.append(float(sum_svm_ent)/n_q)
            kt_svm_ent_orig.append(float(sum_svm_ent_original)/n_q)
        return kt_svm,kt_svm_ent,kt_svm_orig,kt_svm_ent_orig,change_rate_svm,change_rate_svm_ent,range(2,9)

    def calcualte_average_distances(self,competition_data):
        average_distances =[]
        for epoch in competition_data:
            total_average_distance_sum =0
            number_of_queries = 0
            for query in competition_data[epoch]:
                number_of_queries+=1
                sum_distance_query = 0
                denom = 0
                docs = competition_data[epoch][query].keys()
                for doc1,doc2 in itertools.combinations(docs,2):
                    sum_distance_query+=self.cosine_distance(doc1,doc2)
                    denom+=1
                total_average_distance_sum+=float(sum_distance_query)/denom
            average_distances.append(total_average_distance_sum/number_of_queries)
        return average_distances

    def analyze(self,svm,svm_ent,competition_data):
        scores_svm, scores_svm_ent = self.get_all_scores(svm,svm_ent,competition_data)
        rankings_svm_ent, rankings_svm = self.retrieve_ranking(scores_svm, scores_svm_ent)
        kt_svm, kt_svm_ent, kt_svm_orig, kt_svm_ent_orig,change_rate_svm,change_rate_svm_ent, x_axis = self.calculate_average_kendall_tau(rankings_svm_ent, rankings_svm)
        print(kt_svm, kt_svm_ent, kt_svm_orig, kt_svm_ent_orig)
        #average_distances = self.calcualte_average_distances(competition_data)


