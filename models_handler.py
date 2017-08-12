import SVM_SGD as svm_sgd
class models_handler():

    def __init__(self,C_array):
        self.models = []
        for C in C_array:
            self.models.append(svm_sgd.svm_sgd(C))



    def get_right_model_for_fold(self):
        ""
