import analyze_competition as a
import prep as p
import sys
import svm_models_handler as mh
import pickle

import numpy as np
# if __name__=="__main__":
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
analyze = a.analysis()
svm = mh.models_handler([])
model_file = open("svm_model",'rb')

w = pickle.load(model_file)#recover_model("model_light_svm")#pickle.load(model_file)
print(w)
for i in range(1,201):
    svm.query_to_fold_index[i]=1
for i in range(1,6):
    svm.weights_index[i] = w
mhs = [("regular/model_handler_asr_cmp.pickle", 'SVM', 'k')]

mh_svm = preprocess.load_model_handlers(mhs)
# print(mh_svm[0][0].query_to_fold_index)
mh_svm=[(svm,"svm.pickle1","SVM",'k')]
cd = preprocess.extract_features_by_epoch("features_asr_modified")
analyze.compare_rankers(mh_svm,cd)
analyze.create_comparison_plots("results.pickle",svm)