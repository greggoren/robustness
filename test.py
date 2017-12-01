import pickle
model = pickle.load(open("svm_model",'rb'))
print(model)