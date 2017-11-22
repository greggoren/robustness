from models_creation_doubly import SVM_SGD_L1
import operator
import pickle
from models_creation_doubly import params_L1


class single_model_handler_svm_L1():

    def __init__(self, Lambda_array):
        self.models = {}
        for Lambda in Lambda_array:
            self.models[Lambda]=SVM_SGD_L1.svm_sgd_L1(Lambda)


    def fit_model_on_train_set_and_choose_best_for_competition(self,X,y,X_i,y_i,validation_indices,queries,evaluator,preprocess,score_file):
        evaluator.empty_validation_files(params_L1.validation_folder)
        weights = {}
        scores={}
        for Lambda in self.models:
            print("fitting model on Lambda=", Lambda)
            svm = self.models[Lambda]
            svm.fit(X_i,y_i)
            weights[svm.Lambda]=svm.w
            score_file = svm.predict_opt(X, queries, validation_indices,evaluator, score_file,True)
            score = evaluator.run_trec_eval(score_file)
            scores[svm.Lambda] = score
        max_Lambda=max(scores.items(), key=operator.itemgetter(1))[0]
        print("the chosen model is C=",max_Lambda)
        chosen_model = self.models[max_Lambda]
        data_set,tags=preprocess.create_data_set(X, y, queries)
        chosen_model.fit(data_set,tags)

        with open("svm_model_L1.pickle"+str(max_Lambda),'wb') as model_file:
            pickle.dump(chosen_model,model_file)



