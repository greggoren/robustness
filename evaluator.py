
import subprocess

class eval:

    def __init__(self):
        self.metrics = ["map","ndcg_cut.20","P.5","P.10"]

    def create_trec_eval_file(self, test_indices, queries, results,validation=None):#TODO: need to sort file via unix command
        if validation is not None:
            trec_file = validation+"_trec_file.txt"
        else:
            trec_file = "trec_file.txt"
        trec_file_access = open(trec_file,'a')
        for index in test_indices:
            trec_file_access.write(str(queries[index])+"\tQ0\t"+str(index)+"\t"+str(index)+"\t"+str(results[index])+"\tindri\n")
        trec_file_access.close()

    def run_command(self, command):
        p = subprocess.Popen(command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             shell=True)
        return iter(p.stdout.readline, b'')

    def run_trec_eval_on_test_file(self, qrel_path, score_file,model_name):
        score_data = []
        for metric in self.metrics:
            command = "./trec_eval -m " + metric + " " + qrel_path + " " + score_file
            for output_line in self.run_command(command):
                score = output_line.split()[-1]
                score_data.append((model_name, metric, score))

        summary_file = open("summary_of_test_run.txt", 'w')
        summary_file.write("MODEL\tMETRIC\tSCORE\n")
        for score_record in score_data:
            summary_file.write(score_record[0] + "\t" + score_record[1] + "\t" + score_record[2] + "\n")
        summary_file.close()

    def create_qrels_file(self,X,y,queries):
        qrels = open("qrels",'a')
        for i in range(len(X)):
            qrels.write(str(queries[i]) + "\t0\t" + str(i) + "\t" + str(int(y[i])) + "\n")
        qrels.close()