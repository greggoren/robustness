import params
import operator
import subprocess
class single_model_handler_coordinate_ascent():

    def __init__(self,regularization_array):
        self.regularization_param = regularization_array
        self.java_path = "/lv_local/home/sgregory/jdk1.8.0_121/bin/java"
        self.jar_path = "/lv_local/home/sgregory/SEO_CODE/model_running/RankLib.jar"

    def run_bash_command(self,command):
        p = subprocess.Popen(command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, shell=True)
        out, err = p.communicate()
        print(out)
        return out

    def create_model_coordinate_ascent(self, regularization, train_file,
                                query_relevance_file,test=False):

        if test:
            add="test"
        else:
            add=""

        command = self.java_path+' -jar '+self.jar_path+' -train ' + train_file + ' -ranker 4 -qrel ' + query_relevance_file + ' -metric2t NDCG@20' \
                                                               ' -reg ' + str(regularization)+' -save '+add+'model_' + str(regularization)
        print("command = ", command)
        self.run_bash_command(command)

    def run_model(self,test_file,regularization):
        java_path = "/lv_local/home/sgregory/jdk1.8.0_121/bin/java"
        jar_path = "/lv_local/home/sgregory/SEO_CODE/model_running/RankLib.jar"
        score_file = "/lv_local/home/sgregory/robustness/score" + str(regularization)
        features = "/lv_local/home/sgregory/robustness/" + test_file
        model_path = "/lv_local/home/sgregory/robustness/model_"+ str(regularization)
        self.run_bash_command('touch '+score_file)
        command = java_path + " -jar " + jar_path + " -load " + model_path + " -rank " + features + " -score " + score_file
        self.run_bash_command(command)
        return score_file


    def retrieve_scores(self,test_indices,score_file):
        with open(score_file) as scores:
            results={test_indices[i]:score.split()[2].rstrip() for i,score in enumerate(scores)}
            return results



    def fit_model_on_train_set_and_choose_best_for_competition(self,train_file,test_file,validation_indices,queries,evaluator):
        evaluator.empty_validation_files()
        scores={}
        for regularzation in self.regularization_param:
            print("fitting model on ",regularzation)
            self.create_model_coordinate_ascent(regularzation,train_file,params.qrels)
            score_file=self.run_model(test_file,regularzation)
            results = self.retrieve_scores(validation_indices,score_file)
            trec_file=evaluator.create_trec_eval_file(validation_indices, queries, results, "model_"+str(regularzation), validation=True)
            ordered_trec_file = evaluator.order_trec_file(trec_file)
            score = evaluator.run_trec_eval(ordered_trec_file)
            scores[regularzation] = score
        max_regularization=max(scores.items(), key=operator.itemgetter(1))[0]
        print("the chosen model is regilarization:",max_regularization)
        self.create_model_coordinate_ascent(max_regularization,params.data_set_file,params.qrels,True)





