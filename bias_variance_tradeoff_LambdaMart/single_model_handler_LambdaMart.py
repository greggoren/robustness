import os
import subprocess


class single_model_handler_LambdaMart():
    def __init__(self, leaves, trees):
        self.leaves_param = leaves
        self.trees_param = trees
        self.java_path = "/lv_local/home/sgregory/jdk1.8.0_121/bin/java"
        self.jar_path = "/lv_local/home/sgregory/SEO_CODE/model_running/RankLib.jar"

    def run_bash_command(self, command):
        p = subprocess.Popen(command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, shell=True)
        out, err = p.communicate()
        print(out)
        return out

    def create_model_LambdaMart(self, number_of_leaves, train_file,
                                query_relevance_file, number_of_trees):
        models_path = "/lv_local/home/sgregory/robustness/bias_variance_tradeoff_LambdaMart/models/"
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        command = self.java_path + ' -jar ' + self.jar_path + ' -train ' + train_file + ' -ranker 6 -qrel ' + query_relevance_file + ' -metric2t NDCG@20' \
                                                                                                                                     ' -tree ' + str(
            number_of_trees) + ' -leaf ' + str(number_of_leaves) + ' -save ' + models_path + 'model_' + str(
            number_of_trees) + "_" + str(number_of_leaves)
        print("command = ", command)
        self.run_bash_command(command)

    def run_model(self, test_file, trees, leaves, ):
        models_path = "/lv_local/home/sgregory/robustness/bias_variance_tradeoff_LambdaMart/models/"
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        java_path = "/lv_local/home/sgregory/jdk1.8.0_121/bin/java"
        jar_path = "/lv_local/home/sgregory/SEO_CODE/model_running/RankLib.jar"
        score_file = "/lv_local/home/sgregory/robustness/score" + str(trees) + "_" + str(leaves)
        features = "/lv_local/home/sgregory/robustness/" + test_file
        model_path = models_path + "model_" + str(trees) + "_" + str(leaves)
        self.run_bash_command('touch ' + score_file)
        command = java_path + " -jar " + jar_path + " -load " + model_path + " -rank " + features + " -score " + score_file
        self.rund_bash_command(command)
        return score_file

    def retrieve_scores(self, test_indices, score_file):
        with open(score_file) as scores:
            results = {test_indices[i]: score.split()[2].rstrip() for i, score in enumerate(scores)}
            return results
