qrels = "qrels"
validation_folder = "C:/study/svm_test/validation_ent/"
score_file = "C:/study/svm_test/trec_file_asr_ent.txt"
recovery =False
data_set_file ="data/featuresCB_asr"
number_of_folds=5
normalized = True
number_of_competitors = 10
summary_file = "ent_opt.txt"
random_seed = 9001
iter_factor = 1
processes_number = 9
model_handler_file = "model_handler_ent_opt.pickle"
gammas = [0.01,0.001,0.0001]