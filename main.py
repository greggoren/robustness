import analyze_competition as a
import prep as p
import sys

if __name__=="__main__":

    dump =False
    if sys.argv[1]=="R":
        dump=True

    preprocess = p.preprocess()
    analyze = a.analysis()

    # mhs= [("model_handler_ent_opt_minus.pickle0.001",'0.001','b'),("model_handler_ent_opt_minus.pickle0.01",'0.01','g'),("model_handler_ent_opt_minus.pickle0.1",'0.1','r'),("model_handler_ent_opt_minus.pickle0.2",'0.2','m'),("data/model_handler_asr_cmp.pickle",'svm','k')]
    # mhs= [("model_handler_ent_opt_shrinked_minus.pickle0.001",'0.001','b'),("model_handler_ent_opt_shrinked_minus.pickle0.01",'0.01','g'),("model_handler_ent_opt_shrinked_minus.pickle0.1",'0.1','r'),("model_handler_ent_opt_shrinked_minus.pickle0.2",'0.2','m'),("data/model_handler_asr_cmp.pickle",'svm','k')]
    # mhs= [("model_handler_pos_minus.pickle0.001",'0.001','b'),("model_handler_pos_minus.pickle0.01",'0.01','g'),("model_handler_pos_minus.pickle0.1",'0.1','r'),("model_handler_pos_minus.pickle0.2",'0.2','m'),("data/model_handler_asr_cmp.pickle",'svm','k')]
    # mhs= [("abs_models/model_handler_abs.pickle0.1",'0.1','b'),("abs_models/model_handler_abs.pickle0.2",'0.2','g'),("abs_models/model_handler_abs.pickle0.01",'0.01','r'),("abs_models/model_handler_abs.pickle1.0",'1.0','m'),("abs_models/model_handler_abs.pickle10.0",'10.0','c'),("data/model_handler_asr_cmp.pickle",'svm','k')]
    # mhs= [("model_handler_ent_opt_plus.pickle0.3",'0.3','b'),("model_handler_ent_opt_plus.pickle0.5",'0.5','g'),("model_handler_ent_opt_plus.pickle0.7",'0.1','r'),("model_handler_ent_opt_plus.pickle1.0",'1.0','m'),("data/model_handler_asr_cmp.pickle",'svm','k')]
    mhs = [("regular/model_handler_asr_cmp.pickle", 'svm', 'k'),
                 ("regular/model_handler_asr_cmp.pickle", 'svm_epsilon_1', 'y'),("regular/model_handler_asr_cmp.pickle", 'svm_epsilon_1.25', 'm'),("regular/model_handler_asr_cmp.pickle", 'svm_epsilon_1.5', 'g'),("regular/model_handler_asr_cmp.pickle", 'svm_epsilon_2', 'c')]
    # mhs= [("model_handler_pos__sh1.pickle0.001",'0.001','b'),("model_handler_pos__sh1.pickle0.01",'0.01','g'),("model_handler_pos__sh1.pickle0.1",'0.1','r'),("model_handler_pos__sh1.pickle0.2",'0.2','m'),("data/model_handler_asr_cmp.pickle",'svm','k')]
    # mhs= [("model_handler_ent_opt_shrinked1.pickle0.001",'0.001','b'),("model_handler_ent_opt_shrinked1.pickle0.01",'0.01','g'),("model_handler_ent_opt_shrinked1.pickle0.1",'0.1','r'),("model_handler_ent_opt_shrinked1.pickle0.2",'0.2','m'),("data/model_handler_asr_cmp.pickle",'svm','k')]
    mh_svm = preprocess.load_model_handlers(mhs)
    cd = preprocess.extract_features_by_epoch("data/features_asr")
    analyze.analyze(mh_svm,cd,dump)
