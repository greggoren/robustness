import analyze_competition as a
import prep as p
import sys
import svm_models_handler as mh
import pickle


# if __name__=="__main__":

preprocess = p.preprocess()
analyze = a.analysis()
svm = mh.models_handler([])
model_file = open("svm_model",'rb')
w = pickle.load(model_file)
# svm.chosen_model_per_fold[1]=w
for i in range(1,201):
    svm.query_to_fold_index[i]=1
for i in range(1,6):
    svm.weights_index[i] = w
mhs = [("regular/model_handler_asr_cmp.pickle", 'SVM', 'k')]

mh_svm = preprocess.load_model_handlers(mhs)
print(mh_svm[0][0].query_to_fold_index)
mh_svm=[(svm,"svm.pickle1","SVM",'k')]
cd = preprocess.extract_features_by_epoch("features_asr_modified")
#analyze.analyze(mh_svm,cd,dump)
#analyze.create_table_for_epsilons(mh_svm,cd)
#analyze.create_rbo_baseline(mh_svm,cd)
analyze.compare_rankers(mh_svm,cd)
# analyze.create_comparison_plots("results.pickle")
# analyze.get_score_for_LambdaMart_on_clueWeb("data/featuresCB_asr")
