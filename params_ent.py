qrels = "qrels"
validation_folder = "/lv_local/home/sgregory/robustness/validation_ent/"
score_file = "/lv_local/home/sgregory/robustness/trec_file_asr_ent.txt"
recovery =False
data_set_file ="featuresCB_asr"
number_of_folds=5
normalized = False
number_of_competitors = 10
summary_file = "ent_asr.txt"
random_seed = 9001
iter_factor = 1
processes_number = 9
model_handler_file = "model_handler_enr_asr.pickle"
gammas = [0.1,0.01,0.001,0.0001]