import os
import preprocess_clueweb as pc
class preprocess_letor(pc.preprocess):

    def __init__(self,data_set_location):
        self.data_set_location = data_set_location


    def retrieve_sets_for_fold(self):
        for roots,dirs,files in os.walk(self.data_set_location):
            folder = roots[0]+"/"+dirs[0]
            #TODO: add data set retrieval functionality



    