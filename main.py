import analyze_competition as a
import prep as p

preprocess = p.preprocess()
analyze = a.analysis()
mh_svm,mh_svm_ent  = preprocess.load_model_handlers("/lv_local/home/sgregory/robustness/model_handler_asr_cmp.pickle","/lv_local/home/sgregory/robustness/model_handler_ent_asr.pickle")
cd = preprocess.extract_features_by_epoch("features_asr")
analyze.analyze(cd,mh_svm,mh_svm_ent)