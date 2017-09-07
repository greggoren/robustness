import preprocess_clueweb as pc
import pickle
class preprocess(pc.preprocess):
    def __init__(self):
        ""

    def create_index_to_doc_name_dict(self,data_set):
        doc_name_index = {}
        index =0
        with open(data_set) as ds:
            for line in ds:
                rec = line.split("# ")
                doc_name = rec[1].rstrip()
                doc_name_index[index]=doc_name
                index+=1
        return doc_name_index


    def extract_features_by_epoch(self,data_set):
        doc_name_index = self.create_index_to_doc_name_dict(data_set)#TODO:chnage the assignment as an outside param
        competition_data = {}
        X,y,queries = self.retrieve_data_from_file(data_set,True)
        for index in doc_name_index:
            doc_name = doc_name_index[index]
            details = doc_name.split("-")
            epoch = int(details[1])
            if epoch==0:
                continue
            qid = details[2]
            doc = details[3]
            if not competition_data.get(epoch,False):
                competition_data[epoch]={}
            if not competition_data[epoch].get(qid,False):
                competition_data[epoch][qid]={}
            competition_data[epoch][qid][doc] = X[index]
        return competition_data

    def load_model_handlers(self,svms):
        handlers=[]
        for svm in svms:
            handlers.append((pickle.load(open(svm[0],'rb')),svm[0],svm[1],svm[2]))
        return handlers