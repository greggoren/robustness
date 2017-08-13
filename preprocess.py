import os
import numpy as np
import itertools
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GroupKFold
import math

class preprocess:



    def __init__(self,data_set_location):
        self.data_set_location = data_set_location


    def retrieve_data_from_file(self,file):
        print ("loading svmLight file")
        X, y, groups= load_svmlight_file(file,query_id=True)
        print ("loading complete")
        return X.toarray(),y,groups

    def create_data_set(self,X,y,groups):
        print("creating data set")
        data = []
        labels = []
        k=0
        unique_groups = set(groups)
        for group in unique_groups:
            relevant_indexes = np.where(groups==group)[0]
            comb = itertools.combinations(relevant_indexes, 2)
            for (i,j) in comb:
                if (y[i]==y[j]):
                    continue
                data.append(X[i]-X[j])
                labels.append(np.sign(y[i]-y[j]))
                if labels[-1] != (-1) ** k:#to get a balanced data set
                    labels[-1] *= -1
                    data[-1] *= -1
                k += 1
        print ("finished data set creation")
        print ("number of points",len(data))
        return data,labels


    def create_folds(self,X,y,groups,number_of_folds):
        kf = GroupKFold(number_of_folds)
        return kf.split(X,y,groups)



    def create_validation_set(self,number_of_folds,already_been_in_validation_indices,train_indices,number_of_queries,queries):
        validation_queries = set()
        number_of_queries_in_set = math.floor(float(float(number_of_queries)/number_of_folds))
        working_set = train_indices - already_been_in_validation_indices

        validation_set = set()
        for index in working_set:

            if len(validation_queries)>=number_of_queries_in_set:
                break
            validation_queries.add(queries[index])
            validation_set.add(index)
        already_been_in_validation_indices= already_been_in_validation_indices.union(validation_set)
        train_set = train_indices - validation_set
        return already_been_in_validation_indices,validation_set,train_set



    """def index_features_for_competitors(self,normalized):
            feature_index_query = {}
            labels_index = {}

            print "features index creation started"
            amount = 0
            if (normalized):
                amount = 1
            label_index ={}
            #line=1
            for dirs in os.walk(self.data_set_location):
                if dirs[1]:
                    first_directory = dirs[0]+"/"+dirs[1][0]
                    for files in os.walk(first_directory):
                        for file_name in files[2]:
                            current_file = files[0]+"/"+file_name
                            with open(current_file) as features:
                                for feature in features:
                                    if line>=1000:
                                        break
                                    line+=1
                                    feature_data = feature.split()
                                    qid = feature_data[1]
                                    if not feature_index_query.get(qid,False):
                                        feature_index_query[qid]=[]
                                        label_index[qid]=0
                                        labels_index[qid]={}
                                    features_length = len(feature_data)
                                    features_vec = []
                                    for index in range(2, features_length - 1 - amount):
                                        data = feature_data[index]
                                        features_vec.append(float(data.split(":")[1]))
                                    labels_index[qid][label_index[qid]]=int(feature_data[0])
                                    label_index[qid] = label_index[qid]+1
                                    feature_index_query[qid].append(np.array(features_vec))
                print "feature index creation ended"
                return feature_index_query,labels_index"""


    """def create_data_set_svm_rank(self, feature_index_query, labels_index):
        print "data set creation started",
        k = 0
        data_set = []
        labels = []
        transitivity_bigger = {}
        transitivity_smaller = {}
        for qid in feature_index_query:
            if not transitivity_bigger.get(qid, False):
                transitivity_bigger[qid] = {}
            if not transitivity_smaller.get(qid, False):
                transitivity_smaller[qid] = {}
            print "working on ", qid
            comb = itertools.combinations(range(len(feature_index_query[qid])), 2)
            for (i, j) in comb:

                if transitivity_bigger[qid].get(i, None) is None:
                    transitivity_smaller, transitivity_bigger = self.initialize_edges(transitivity_smaller,
                                                                                      transitivity_bigger, i, qid)
                if transitivity_bigger[qid].get(j, None) is None:
                    transitivity_smaller, transitivity_bigger = self.initialize_edges(transitivity_smaller,
                                                                                      transitivity_bigger, j, qid)
                if labels_index[qid][i] == labels_index[qid][j]:
                    continue

                sign = np.sign(labels_index[qid][i] - labels_index[qid][j])

                if sign == -1:
                    transitivity_smaller[qid][i].add(j)
                    transitivity_bigger[qid][j].add(i)
                    if self.check_transitivity(transitivity_bigger[qid][j], transitivity_smaller[qid][i]):
                        continue
                else:
                    transitivity_smaller[qid][j].add(i)
                    transitivity_bigger[qid][i].add(j)
                    if self.check_transitivity(transitivity_bigger[qid][i], transitivity_smaller[qid][j]):
                        continue

                data_set.append(feature_index_query[qid][i] - feature_index_query[qid][j])
                labels.append(sign)
                if labels[-1] != (-1) ** k:
                    labels[-1] *= -1
                    data_set[-1] *= -1
                k += 1
        print len(labels)
        print "data set creation ended"
        del (feature_index_query)

        return data_set, labels"""


    """def initialize_edges(self, smaller, bigger, k, qids):
        smaller[qids][k] = set()
        bigger[qids][k] = set()
        return smaller, bigger


    def check_transitivity(self, bigger_set, smaller_set):
        a = smaller_set.intersection(bigger_set)
        if len(a) != 0:
            return True
        return False"""
