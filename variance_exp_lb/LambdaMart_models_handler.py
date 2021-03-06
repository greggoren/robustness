import params
import operator
import subprocess
import os


class model_handler_LambdaMart():
    def __init__(self, trees, leaves):
        self.leaf_number = leaves
        self.trees_number = trees
        self.java_path = "/lv_local/home/sgregory/jdk1.8.0_121/bin/java"
        self.jar_path = "/lv_local/home/sgregory/SEO_CODE/model_running/RankLib.jar"
        self.query_to_fold_index = {}
        self.chosen_model_per_fold = {}

    def run_bash_command(self, command):
        p = subprocess.Popen(command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, shell=True)
        out, err = p.communicate()
        # print(out)
        return out

    def set_queries_to_folds(self, queries, test_indices, fold):
        set_of_queries = set(queries[test_indices])
        tmp = {a: fold for a in set_of_queries}
        self.query_to_fold_index.update(tmp)

    def create_model_LambdaMart(self, number_of_trees, number_of_leaves, train_file,
                                fold, qrels, subset, test=False):

        if test:
            add = "test"
        else:
            add = ""
        try:
            if not os.path.exists("models/" + str(
                    fold)):
                os.makedirs("models/" + str(fold))
        except:
            print("collision on models dir")
        command = self.java_path + ' -jar ' + self.jar_path + ' -train ' + train_file + ' -ranker 6 -qrel ' + qrels + ' -metric2t NDCG@20' \
                                                                                                                      ' -tree ' + str(
            number_of_trees) + ' -leaf ' + str(number_of_leaves) + ' -save ' + "models/" + str(
            fold) + "/" + add + 'model_' + str(number_of_trees) + "_" + str(number_of_leaves) + str(subset)
        print("command = ", command)
        self.run_bash_command(command)

    def run_model(self, test_file, fold, trees, leaves, subset):
        java_path = "/lv_local/home/sgregory/jdk1.8.0_121/bin/java"
        jar_path = "/lv_local/home/sgregory/SEO_CODE/model_running/RankLib.jar"
        score_file = "scores/" + str(fold) + "/score" + str(trees) + "_" + str(
            leaves) + "#" + str(subset)
        try:
            if not os.path.exists("scores/" + str(fold) + "/"):
                os.makedirs("scores/" + str(fold) + "/")
        except:
            print("collision on scores dir")
        features = test_file
        model_path = "models/" + str(fold) + "/model_" + str(trees) + "_" + str(
            leaves) + str(subset)
        self.run_bash_command('touch ' + score_file)
        command = java_path + " -jar " + jar_path + " -load " + model_path + " -rank " + features + " -score " + score_file
        self.run_bash_command(command)
        return score_file

    def run_model_on_test(self, test_file, fold, trees, leaves):
        java_path = "/lv_local/home/sgregory/jdk1.8.0_121/bin/java"
        jar_path = "/lv_local/home/sgregory/SEO_CODE/model_running/RankLib.jar"
        score_file = "/lv_local/home/sgregory/robustness/scores/" + str(fold) + "/score" + "_" + str(fold) + "_" + str(
            trees) + "_" + str(leaves)
        if not os.path.exists("/lv_local/home/sgregory/robustness/relecance_check_LambdaMart/scores/" + str(
                fold) + "/"):
            os.makedirs("/lv_local/home/sgregory/robustness/relecance_check_LambdaMart/scores/" + str(fold) + "/")
        features = test_file
        model_path = "/lv_local/home/sgregory/robustness/relecance_check_LambdaMart/models/" + str(
            fold) + "/model_" + str(trees) + "_" + str(
            leaves)
        self.run_bash_command('touch ' + score_file)
        command = java_path + " -jar " + jar_path + " -load " + model_path + " -rank " + features + " -score " + score_file
        self.run_bash_command(command)
        return score_file

    def retrieve_scores(self, test_indices, score_file):
        with open(score_file) as scores:
            results = {test_indices[i]: score.split()[2].rstrip() for i, score in enumerate(scores)}
            return results

    def fit_model_on_train_set_and_run(self, train_file, test_file, test_indices, queries, evaluator, qrels, fold):
        print("fitting model on trees=", self.trees_number, "leaves = ", self.leaf_number)
        self.create_model_LambdaMart(self.trees_number, self.leaf_number, train_file, fold, qrels)
        score_file = self.run_model(test_file, fold, self.trees_number, self.leaf_number)
        results = self.retrieve_scores(test_indices, score_file)
        trec_file = evaluator.create_trec_eval_file(test_indices, queries, results,
                                                    "model_" + str(self.trees_number) + "_" + str(self.leaf_number),
                                                    validation=False)
        return trec_file

    def fit_model_on_train_set_for_variance(self, qrels, fold, subset):
        train_file = "train/" + str(fold) + "/features" + str(subset)
        test_file = "test/" + str(fold) + "/features0_test"
        print("fitting model on trees=", self.trees_number, "leaves = ", self.leaf_number)
        self.create_model_LambdaMart(self.trees_number, self.leaf_number, train_file, fold, qrels, subset)
        score_file = self.run_model(test_file, fold, self.trees_number, self.leaf_number, subset)
        return score_file
