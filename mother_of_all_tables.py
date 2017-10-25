import os
import analyze_competition as a
import prep as p
def create_mhs(dir):
    mhs = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            file_name = root+"/"+file
            model = file.split("pickle")[1]
            mhs.append((file_name,model,'a'))
    return mhs


if __name__=="__main__":
    preprocess = p.preprocess()
    analyze = a.analysis()
    meta_mhs = []
    name_dict = {"pos_plus":"POS/NEG Max","pos_plus_big":"POS/NEG Max","pos_minus_big":"POS/NEG Min",'pos_minus':"POS/NEG Min",'squared_minus_big':"Squared Min",'squared_plus_big':"Squared Max",'squared_minus':"Squared Min",'squared_plus':"Squared Max","regular":"SVM"}
    # dirs = ['pos_plus','pos_minus','squared_minus','squared_plus']
    dirs = ['pos_plus','pos_minus','squared_plus','squared_minus']
    scores_dict = analyze.read_retrieval_scores("ret_res")
    for dir in dirs:
        meta_mhs.append(create_mhs(dir))
    meta_model_objects = []
    baselines = [("regular/model_handler_asr_cmp.pickle",'svm','k'),("regular/model_handler_asr_cmp.pickle",'svm_epsilon','k')]
    baselines_model_objects=preprocess.load_model_handlers(baselines)
    for mhs in meta_mhs:
        meta_model_objects.append(preprocess.load_model_handlers(mhs))
    #baseline = preprocess.load_model_handlers([create_mhs("regular")][0])
    cd = preprocess.extract_features_by_epoch("data/features_asr")
    #analyze.create_table(meta_model_objects,cd,name_dict,scores_dict,baseline,baselines_model_objects)
    analyze.get_metrices_for_table(meta_model_objects,cd)