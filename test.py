import pickle
model = pickle.load(open("test/svm_model_minmax.pickle0.001_10.0_10.0",'rb'))
print(model.w)