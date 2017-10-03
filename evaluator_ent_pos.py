import shutil
import subprocess
import os
import params_ent_shrinked_pos as params_ent
class eval:




    def __init__(self):
        self.metrics = ["map","ndcg_cut.20","P.5","P.10"]
        self.validation_metric = "ndcg_cut.20"
        self.doc_name_index = {}

    def remove_score_file_from_last_run(self,score_file):
        command = "rm "+score_file
        for line in self.run_command(command):
            print(line)

    def create_trec_eval_file(self, test_indices, queries, results,model,validation=None):#TODO: need to sort file via unix command
        if validation is not None:
            trec_file = params_ent.validation_folder+"/trec_file_"+model+".txt"
            if not os.path.exists(os.path.dirname(trec_file)):
                os.makedirs(os.path.dirname(trec_file))

        else:
            trec_file = params_ent.score_file
        trec_file_access = open(trec_file,'a')
        for index in test_indices:
            trec_file_access.write(self.set_qid_for_trec(queries[index])+" Q0 "+self.doc_name_index[index]+" "+str(0)+" "+str(results[index])+" seo\n")
        trec_file_access.close()
        return trec_file



    def create_trec_eval_file_opt(self, test_indices, queries, results,model,score,gamma,sigma,validation=None):
        if validation is not None:
            trec_file = params_ent.validation_folder+str(sigma)+"_"+str(gamma)+"/trec_file_"+model+".txt"
            if not os.path.exists(os.path.dirname(trec_file)):
                os.makedirs(os.path.dirname(trec_file))

        else:
            trec_file = score
        trec_file_access = open(trec_file,'a')
        for index in test_indices:
            trec_file_access.write(self.set_qid_for_trec(queries[index])+" Q0 "+self.doc_name_index[index]+" "+str(0)+" "+str(results[index])+" seo\n")
        trec_file_access.close()
        return trec_file


    def run_command(self, command):
        p = subprocess.Popen(command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             shell=True)
        return iter(p.stdout.readline, b'')

    def run_trec_eval(self, score_file):
        command = "./trec_eval -m " + self.validation_metric + " " + params_ent.qrels + " " + score_file
        for output_line in self.run_command(command):
            print("output line=",output_line)
            score = output_line.split()[-1].rstrip()
            break
        return score

    def empty_validation_files(self,validation_folder):
        try:
            shutil.rmtree(validation_folder)
        except:
            print("no validation folder")


    def run_trec_eval_on_test(self):
        score_data = []
        print("last stats:")
        for metric in self.metrics:
            command = "./trec_eval -m " + metric + " " + params_ent.qrels + " " + params_ent.score_file
            for output_line in self.run_command(command):
                print(metric,output_line)
                score = output_line.split()[-1].rstrip()
                score_data.append((metric, str(score)))
        summary_file = open(params_ent.summary_file, 'w')
        summary_file.write("METRIC\tSCORE\n")
        for score_record in score_data:
            next_line = score_record[0] + "\t" + score_record[1] + "\n"
            summary_file.write(next_line)
        summary_file.close()

    def run_trec_eval_on_test_for_opt(self,score_file,summary):
        score_data = []
        print("last stats:")
        for metric in self.metrics:
            command = "./trec_eval -m " + metric + " " + params_ent.qrels + " " + score_file
            for output_line in self.run_command(command):
                print(metric,output_line)
                score = output_line.split()[-1].rstrip()
                score_data.append((metric, str(score)))
        summary_file = open(summary, 'w')
        summary_file.write("METRIC\tSCORE\n")
        for score_record in score_data:
            next_line = score_record[0] + "\t" + score_record[1] + "\n"
            summary_file.write(next_line)
        summary_file.close()

    def create_index_to_doc_name_dict(self):
        index =0
        with open(params_ent.data_set_file) as ds:
            for line in ds:
                rec = line.split("# ")
                doc_name = rec[1].rstrip()
                self.doc_name_index[index]=doc_name
                index+=1

    def set_qid_for_trec(self,query):
        if query < 10:
            qid = "00" + str(query)
        elif query < 100:
            qid = "0" + str(query)
        else:
            qid = str(query)
        return qid

    def create_qrels_file(self,X,y,queries):
        print("creating qrels file")
        qrels = open(params_ent.qrels,'w')
        for i in range(len(X)):
            qrels.write(self.set_qid_for_trec(queries[i]) + " 0 " + self.doc_name_index[i] + " " + str(int(y[i])) + "\n")
        qrels.close()
        print("qrels file ended")
