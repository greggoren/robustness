from models_creation_minmax import SVM_SGD_ENT_POS_MINMAX as svm_sgd_entropy_pos_minmax
import operator
import pickle
from models_creation_minmax import params_ent_pos_minmax
class single_model_handler_svm_entropy_minmax():
    def __init__(self,C_array,Gamma_array,Sigma_array):
        self.models = {}
        for C in C_array:
            for Gamma in Gamma_array:
                for Sigma in Sigma_array:
                    self.models[(C,Gamma,Sigma)]=svm_sgd_entropy_pos_minmax.svm_sgd_entropy_pos_minmax(C,Gamma,Sigma)


    def fit_model_on_train_set_and_choose_best_for_competition(self,X,y,X_i,y_i,validation_indices,queries,evaluator,preprocess,score_file):
        evaluator.empty_validation_files(params_ent_pos_minmax.validation_folder)
        weights = {}
        scores={}
        for C,Gamma,Sigma in self.models:
            print("fitting model on C=", C," Gamma=",Gamma," Sigma=",Sigma)
            svm = self.models[(C,Gamma,Sigma)]
            svm.fit(X_i,y_i)
            weights[svm.C]=svm.w
            score_file = svm.predict_opt(X, queries, validation_indices,evaluator, score_file,True)
            score = evaluator.run_trec_eval(score_file)
            scores[(svm.C,svm.Gamma,svm.Sigma)] = score
        max_C,max_Gamma,max_Sigma=max(scores.items(), key=operator.itemgetter(1))[0]
        print("the chosen model is C=",str(max_C)," Gamma=",max_Gamma," Sigma=",max_Sigma)
        chosen_model = self.models[(max_C,max_Gamma,max_Sigma)]
        data_set,tags=preprocess.create_data_set(X, y, queries)
        chosen_model.fit(data_set,tags)

        with open("svm_model_minmax.pickle"+str(max_C)+"_"+str(max_Gamma)+"_"+str(max_Sigma),'wb') as model_file:
            pickle.dump(chosen_model,model_file)



