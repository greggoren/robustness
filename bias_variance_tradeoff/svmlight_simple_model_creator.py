from bias_variance_tradeoff import params
import numpy as np
import os
import subprocess
import svm_models_handler as mh


def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    return iter(p.stdout.readline, b'')


def learn_svm(C, train_file):
    if not os.path.exists("./models_light/"):
        os.makedirs("./models_light/")
    model_file = "./models_light/svm_model" + str(C)
    learning_command = "./svm_rank_learn -c " + str(C) + " " + train_file + " " + model_file
    for output_line in run_command(learning_command):
        print(output_line)
    return model_file




if __name__ == "__main__":
    C_array = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
    train_file = params.data_set_file
    for C in C_array:
        model_file = learn_svm(C, train_file)

    C_array = [float(i) / 100 for i in range(1, 10)]
    for C in C_array:
        model_file = learn_svm(C, train_file)

    C_array = [float(i) / 10 for i in range(1, 10)]
    for C in C_array:
        model_file = learn_svm(C, train_file)

    C_array = [float(i) for i in range(1, 10)]
    for C in C_array:
        model_file = learn_svm(C, train_file)

    C_array = [float(i * 100) for i in range(1, 10)]
    for C in C_array:
        model_file = learn_svm(C, train_file)

    C_array = [float(i * 1000) for i in range(1, 10)]
    for C in C_array:
        model_file = learn_svm(C, train_file)
