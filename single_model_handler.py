import SVM_SGD as svm_sgd
import evaluator as e
import operator
import pickle
class single_model_handler():

    def __init__(self,C_array):
        self.models = {}
        for C in C_array:
            self.models[C]=svm_sgd.svm_sgd(C)


    def fit_model_on_train_set_and_choose_best_for_competition(self,X,y,X_i,y_i,validation_indices,queries,evaluator,preprocess):
        evaluator.empty_validation_files()
        weights = {}
        scores={}
        for C in self.models:
            print("fitting model on", C)
            svm = self.models[C]
            svm.fit(X_i,y_i)
            weights[svm.C]=svm.w
            score_file = svm.predict(X, queries, validation_indices,evaluator, True)
            score = evaluator.run_trec_eval(score_file)
            scores[svm.C] = score
        max_C=max(scores.items(), key=operator.itemgetter(1))[0]
        print("the chosen model is",str(max_C))
        chosen_model = self.models[max_C]
        data_set,tags=preprocess.create_data_set(X, y, queries)
        chosen_model.fit(data_set,tags)
        with open("svm_model",'wb') as model_file:
            pickle.dump(chosen_model,model_file)


    def predict(self,X,queries,test_indices,fold,eval):
        svm = svm_sgd.svm_sgd(C=self.chosen_model_per_fold[fold])
        svm.w = self.weights_index[fold]
        svm.predict(X,queries,test_indices,eval)


