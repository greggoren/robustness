from models_creation_doubly_regularized_C import SVM_SGD_doubly
import operator
import pickle
from models_creation_doubly_regularized_C import params_doubly


class single_model_handler_svm_doubly():

    def __init__(self, Lambda1_array,Lambda2_array,C_array):
        self.models = {}
        for Lambda1 in Lambda1_array:
            for Lambda2 in Lambda2_array:
                for C in C_array:
                    self.models[(Lambda1,Lambda2,C)]=SVM_SGD_doubly.svm_sgd_doubly(Lambda1,Lambda2,C)


    def fit_model_on_train_set_and_choose_best_for_competition(self,X,y,X_i,y_i,validation_indices,queries,evaluator,preprocess,score_file):
        evaluator.empty_validation_files(params_doubly.validation_folder)
        weights = {}
        scores={}
        for Lambda1,Lambda2,C in self.models:
            print("fitting model on Lambda1=", Lambda1,"Lambda2=",Lambda2,"C=",C)
            svm = self.models[(Lambda1,Lambda2,C)]
            svm.fit(X_i,y_i)
            weights[(svm.Lambda1,svm.Lambda2,svm.C)]=svm.w
            score_file = svm.predict_opt(X, queries, validation_indices,evaluator, score_file,True)
            ordered_trec_file = evaluator.order_trec_file(score_file)
            score = evaluator.run_trec_eval(ordered_trec_file)
            scores[(svm.Lambda1,svm.Lambda2,svm.C)] = score
        max_Lambda1,max_Lambda2,max_C=max(scores.items(), key=operator.itemgetter(1))[0]
        print("the chosen model is Lambda1=", max_Lambda1,"Lambda2=",max_Lambda2,"C=",max_C)
        chosen_model = self.models[(max_Lambda1,max_Lambda2,max_C)]
        data_set,tags=preprocess.create_data_set(X, y, queries)
        chosen_model.fit(data_set,tags)
        print("chosen weights=", [str(round(a, 3)) for a in chosen_model.w])

        with open("svm_model_doubly.pickle"+str(max_C)+"_"+str(max_Lambda1)+"_"+str(max_Lambda2),'wb') as model_file:
            pickle.dump(chosen_model,model_file)



