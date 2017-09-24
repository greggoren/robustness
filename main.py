import analyze_competition as a
import prep as p
import sys

if __name__=="__main__":

    dump =False
    if sys.argv[1]=="R":
        dump=True

    preprocess = p.preprocess()
    analyze = a.analysis()

    mhs= [("model_handler_ent_opt_minus.pickle0.001",'0.001','b'),("model_handler_ent_opt_minus.pickle0.01",'0.01','g'),("model_handler_ent_opt_minus.pickle0.1",'0.1','r'),("model_handler_ent_opt_minus.pickle0.2",'0.2','m'),("data/model_handler_asr_cmp.pickle",'svm','k')]
    # mhs= [("model_handler_ent_opt_shrinked_minus.pickle0.001",'0.001','b'),("model_handler_ent_opt_shrinked_minus.pickle0.01",'0.01','g'),("model_handler_ent_opt_shrinked_minus.pickle0.1",'0.1','r'),("model_handler_ent_opt_shrinked_minus.pickle0.2",'0.2','m'),("data/model_handler_asr_cmp.pickle",'svm','k')]
    # mhs= [("model_handler_pos_minus.pickle0.001",'0.001','b'),("model_handler_pos_minus.pickle0.01",'0.01','g'),("model_handler_pos_minus.pickle0.1",'0.1','r'),("model_handler_pos_minus.pickle0.2",'0.2','m'),("data/model_handler_asr_cmp.pickle",'svm','k')]
    # mhs= [("model_handler_shrinked_pos_minus.pickle0.001",'0.001','b'),("model_handler_shrinked_pos_minus.pickle0.01",'0.01','g'),("model_handler_shrinked_pos_minus.pickle0.1",'0.1','r'),("model_handler_shrinked_pos_minus.pickle0.2",'0.2','m'),("data/model_handler_asr_cmp.pickle",'svm','k')]
    mh_svm = preprocess.load_model_handlers(mhs)
    cd = preprocess.extract_features_by_epoch("data/features_asr")
    analyze.analyze(mh_svm,cd,dump)
