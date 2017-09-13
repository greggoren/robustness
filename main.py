import analyze_competition as a
import prep as p




preprocess = p.preprocess()
analyze = a.analysis()
#mh_svm,mh_svm_ent = preprocess.load_model_handlers("/lv_local/home/sgregory/robustness/model_handler_asr_cmp.pickle","/lv_local/home/sgregory/robustness/model_handler_ent_asr.pickle")
mhs= [("data2/model_handler_ent_opt_sh.pickle0.001",'0.001','b'),("data2/model_handler_ent_opt_sh.pickle0.01",'0.01','g'),("data2/model_handler_ent_opt_sh.pickle0.1",'0.1','r'),("data2/model_handler_ent_opt_sh.pickle0.2",'0.2','m'),("data/model_handler_asr_cmp.pickle",'svm','k')]
mhs= [("data/model_handler_asr_cmp.pickle",'svm','k'),("data/model_handler_asr_cmp.pickle",'svm_epsilon','b')]
# mhs= [("data/model_handler_asr_cmp.pickle",'svm','g'),("data/model_handler_ent_opt_cmb.pickle",'svm ent combined','b')]
mh_svm = preprocess.load_model_handlers(mhs)
cd = preprocess.extract_features_by_epoch("data/features_asr")
analyze.analyze(mh_svm,cd)
