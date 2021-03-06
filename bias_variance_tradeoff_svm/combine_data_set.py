def read_data_sets_and_append(data_sets, i):
    with open("featuresASR_combined" + i, 'w') as combined:
        for data_set in data_sets:
            f = open(data_set)
            for line in f:
                features = line.split(" # ")[0]
                name = line.split(" # ")[1]
                splited_name = name.split("-")
                new_name = splited_name[0] + "-" + splited_name[1] + "-" + splited_name[2].split("_")[0] + "-" + \
                           splited_name[3]
                new_line = features + " # " + new_name
                combined.write(new_line)
            f.close()


data_sets1 = ["../featuresASR_round1_SVM", "../featuresASR_round1_LambdaMART"]
data_sets2 = ["../featuresASR_round2_SVM",
              "../featuresASR_round2_LambdaMART"]
read_data_sets_and_append(data_sets1, "1")
read_data_sets_and_append(data_sets2, "2")
