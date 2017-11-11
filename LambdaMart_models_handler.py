import params
import operator
import subprocess
class model_handler_LambdaMart():

    def __init__(self,leaves,trees):
        self.leaves_param = leaves
        self.trees_param = trees
        self.java_path = "/lv_local/home/sgregory/jdk1.8.0_121/bin/java"
        self.jar_path = "/lv_local/home/sgregory/SEO_CODE/model_running/RankLib.jar"
        self.query_to_fold_index={}
        self.chosen_model_per_fold ={}
    def run_bash_command(self,command):
        p = subprocess.Popen(command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, shell=True)
        out, err = p.communicate()
        # print(out)
        return out

    def set_queries_to_folds(self,queries,test_indices,fold):
        set_of_queries = set(queries[test_indices])
        tmp = {a:fold for a in set_of_queries}
        self.query_to_fold_index.update(tmp)

    def create_model_LambdaMart(self, number_of_trees, number_of_leaves, train_file,
                                query_relevance_file,test=False):

        if test:
            add="test"
        else:
            add=""

        command = self.java_path+' -jar '+self.jar_path+' -train ' + train_file + ' -ranker 6 -qrel ' + query_relevance_file + ' -metric2t NDCG@20' \
                                                               ' -tree ' + str(number_of_trees) + ' -leaf ' + str(number_of_leaves) +' -save '+add+'model_' + str(number_of_trees) + "_" + str(number_of_leaves)
        print("command = ", command)
        self.run_bash_command(command)

    def run_model(self,test_file,trees,leaves):
        java_path = "/lv_local/home/sgregory/jdk1.8.0_121/bin/java"
        jar_path = "/lv_local/home/sgregory/SEO_CODE/model_running/RankLib.jar"
        score_file = "/lv_local/home/sgregory/robustness/score" + str(trees)+"_"+str(leaves)
        features = "/lv_local/home/sgregory/robustness/" + test_file
        model_path = "/lv_local/home/sgregory/robustness/model_"+str(trees)+"_"+str(leaves)
        self.run_bash_command('touch '+score_file)
        command = java_path + " -jar " + jar_path + " -load " + model_path + " -rank " + features + " -score " + score_file
        self.run_bash_command(command)
        return score_file

    def run_model_on_test(self,test_file,fold,trees,leaves):
        java_path = "/lv_local/home/sgregory/jdk1.8.0_121/bin/java"
        jar_path = "/lv_local/home/sgregory/SEO_CODE/model_running/RankLib.jar"
        score_file = "/lv_local/home/sgregory/robustness/score"+"_"+str(fold)+"_" + str(trees)+"_"+str(leaves)
        features = "/lv_local/home/sgregory/robustness/" + test_file
        model_path = "/lv_local/home/sgregory/robustness/model_"+str(trees)+"_"+str(leaves)
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
        for trees_number in self.trees_param:
            for leaf_number in self.leaves_param:
                print("fitting model on trees=", trees_number,"leaves = ",leaf_number)
                self.create_model_LambdaMart(trees_number,leaf_number,train_file,params.qrels)
                score_file=self.run_model(test_file,trees_number,leaf_number)
                results = self.retrieve_scores(validation_indices,score_file)
                trec_file=evaluator.create_trec_eval_file(validation_indices, queries, results, "model_"+str(trees_number)+"_"+str(leaf_number), validation=True)
                score = evaluator.run_trec_eval(trec_file)
                scores[(trees_number,leaf_number)] = score
        trees,leaves=max(scores.items(), key=operator.itemgetter(1))[0]
        print("the chosen model is trees=",trees," leaves=",leaves)
        self.create_model_LambdaMart(trees,leaves,params.data_set_file,params.qrels,True)




    def fit_model_on_train_set_and_choose_best(self,train_file,test_file,validation_indices,queries,fold,query_relevance_file,evaluator):
        print("fitting models on fold",fold)
        scores={}
        for trees_number in self.trees_param:
            for leaf_number in self.leaves_param:
                self.create_model_LambdaMart(trees_number,leaf_number,train_file,query_relevance_file)
                # weights[svm.C]=svm.w
                score_file = self.run_model(test_file,trees_number,leaf_number)
                results = self.retrieve_scores(validation_indices,score_file)
                trec_file=evaluator.create_trec_eval_file(validation_indices,queries,results,"_".join([str(a) for a in (trees_number,leaf_number)]),True)
                score = evaluator.run_trec_eval(trec_file)
                scores[((trees_number,leaf_number))] = score
        trees, leaves = max(scores.items(), key=operator.itemgetter(1))[0]
        print("the chosen model is trees=", trees, " leaves=", leaves)
        self.chosen_model_per_fold[fold]=(trees,leaves)
