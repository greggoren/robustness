from epsilon_picking import analyze as a
import prep as p
import sys
import svm_models_handler as mh
import pickle

import numpy as np
# if __name__=="__main__":

def get_banned(banned_file):
    banned_queries={i:[] for i in [0,1,2,3,4]}
    with open(banned_file) as banned:
        for ban in banned:
            splitted=ban.split()
            banned_queries[int(splitted[0])].append(splitted[1])
    return banned_queries

def recover_model(model):
    indexes_covered = []
    weights =[]
    with open(model) as model_file:
        for line in model_file:
            if line.__contains__(":"):
                wheights = line.split()
                wheights_length = len(wheights)

                for index in range(1, wheights_length - 1):

                    feature_id = int(wheights[index].split(":")[0])
                    if index < feature_id:
                        for repair in range(index, feature_id):
                            if repair in indexes_covered:
                                continue
                            weights.append(0)
                            indexes_covered.append(repair)
                    weights.append(float(wheights[index].split(":")[1]))
                    indexes_covered.append(feature_id)
    return np.array(weights)

preprocess = p.preprocess()
analyze = a.analyze()
svm = mh.models_handler([])
model_file = open("../svm_model",'rb')
banned_queries=get_banned("../banned")
w = pickle.load(model_file)#recover_model("model_light_svm")#pickle.load(model_file)
print(w)
for i in range(1,201):
    svm.query_to_fold_index[str(i).zfill(3)+"_1"]=1
    svm.query_to_fold_index[str(i).zfill(3)+"_2"]=1
    # svm.query_to_fold_index[str(i).zfill(3)]=1
for i in range(1,6):
    svm.weights_index[i] = w

mh_svm=[(svm,"svm.pickle1","SVM",'k')]
cd = preprocess.extract_features_by_epoch("../data/featuresASR_L")

analyze.create_epsilon_for_Lambda_mart(cd,mh_svm,banned_queries)
