import SVM_SGD_ENT as svm_sgd_ent
import evaluator as e
import operator
import sys





class svm_ent_models_handler():
    def __init__(self, C_array,Gamma_array):
        self.models = []
        for C in C_array:
            for Gamma in Gamma_array:
                self.models.append(svm_sgd_ent.svm_sgd_entropy(C,Gamma))

        self.query_to_fold_index = {}
        self.weights_index = {}
        self.chosen_model_per_fold = {}


    def set_queries_to_folds(self, queries, test_indices, fold):
        set_of_queries = set(queries[test_indices])
        tmp = {a: fold for a in set_of_queries}
        self.query_to_fold_index.update(tmp)

    def fit_models(self,X,y,svm):
        svm.fit(X, y)
        return svm

    def fit_model_on_train_set_and_choose_best(self, X, X_i, y_i, validation_indices, fold, queries, evaluator):

        print("fitting models on fold", fold)
        weights = {}
        scores = {}
        for svm in self.models:
            sys.stdout.flush()
            svm.fit(X_i, y_i)
            weights[(svm.C,svm.Gamma)] = svm.w
            score_file = svm.predict(X, queries, validation_indices, evaluator, True)
            score = evaluator.run_trec_eval(score_file)
            scores[(svm.C,svm.Gamma)] = score
        max_C,max_Gamma = max(scores.items(), key=operator.itemgetter(1))[0]
        print("on fold", str(fold), "the chosen model is", str(max_C),str(max_Gamma))
        self.weights_index[fold] = weights[(max_C,max_Gamma)]
        self.chosen_model_per_fold[fold] = (max_C,max_Gamma)

    def fit_model_on_train_set_and_choose_best_opt(self, X, X_i, y_i, validation_indices, fold, queries,score_opt,gamma,evaluator):
        print("fitting models on fold", fold)
        weights = {}
        scores = {}
        for svm in self.models:
            sys.stdout.flush()
            svm.fit(X_i, y_i)
            weights[(svm.C,svm.Gamma)] = svm.w
            score_file = svm.predict_opt(X, queries, validation_indices, evaluator, score_opt,gamma,True)
            score = evaluator.run_trec_eval(score_file)
            scores[(svm.C,svm.Gamma)] = score
        max_C,max_Gamma = max(scores.items(), key=operator.itemgetter(1))[0]
        print("on fold", str(fold), "the chosen model is", str(max_C),str(max_Gamma))
        self.weights_index[fold] = weights[(max_C,max_Gamma)]
        self.chosen_model_per_fold[fold] = (max_C,max_Gamma)

    def predict(self, X, queries, test_indices, fold, eval):
        C,Gamma = self.chosen_model_per_fold[fold]
        svm = svm_sgd_ent.svm_sgd_entropy(C,Gamma)
        svm.w = self.weights_index[fold]
        svm.predict(X, queries, test_indices, eval)

    def predict_opt(self, X, queries, test_indices, fold,score,eval,gamma):
        C,Gamma = self.chosen_model_per_fold[fold]
        svm = svm_sgd_ent.svm_sgd_entropy(C,Gamma)
        svm.w = self.weights_index[fold]
        svm.predict_opt(X, queries, test_indices,eval,score,gamma)
