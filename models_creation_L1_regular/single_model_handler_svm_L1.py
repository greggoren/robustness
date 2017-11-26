from models_creation_L1_regular import SVM_SGD_L1
import operator
import pickle
from models_creation_L1_regular import params_L1


class single_model_handler_svm_L1():

    def __init__(self, Lambda_array,C_array):
        self.models = {}
        for Lambda in Lambda_array:
            for C in C_array:
                self.models[(Lambda,C)]=SVM_SGD_L1.svm_sgd_L1(Lambda,C)


    def fit_model_on_train_set_and_choose_best_for_competition(self,X,y,X_i,y_i,validation_indices,queries,evaluator,preprocess,score_file):
        evaluator.empty_validation_files(params_L1.validation_folder)
        weights = {}
        scores={}
        for Lambda,C in self.models:
            print("fitting model on Lambda=", Lambda,"C=",C)
            svm = self.models[(Lambda,C)]
            svm.fit(X_i,y_i)
            weights[(svm.Lambda,svm.C)]=svm.w
            score_file = svm.predict_opt(X, queries, validation_indices,evaluator, score_file,True)
            ordered_trec_file = evaluator.order_trec_file(score_file)
            score = evaluator.run_trec_eval(ordered_trec_file)
            scores[(svm.Lambda,svm.C)] = score
            print("weights=",[str(round(a,3)) for a in svm.w])
        max_Lambda,max_C=max(scores.items(), key=operator.itemgetter(1))[0]
        print("the chosen model is Lambda=",max_Lambda,"C=",max_C)
        chosen_model = self.models[(max_Lambda,max_C)]
        data_set,tags=preprocess.create_data_set(X, y, queries)
        chosen_model.fit(data_set,tags)

        with open("svm_model_L1.pickle"+str(max_C)+"_"+str(max_Lambda),'wb') as model_file:
            pickle.dump(chosen_model,model_file)



