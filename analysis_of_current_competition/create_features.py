def append_features(epochs):
    features = open("featuresASR_round2", 'w')
    for e in epochs:
        file = "epoch"+str(e)+"/features_ww"
        with open(file) as features_file:
            for feature in features_file:
                feature_parts=feature.split(" # ")
                name = "ROUND-" + str(e - 5).zfill(2) + "-" + feature_parts[1].split("-")[0] + "-ObjectId(" + \
                       feature_parts[1].split("-")[1].rstrip() + ")"


                new_line = feature.split(" # ")[0]+" # "+name+"\n"
                features.write(new_line)
    features.close()


# epochs = [1, 2, 3, 4, 5]
epochs = [6, 7, 8, 9, 10]

append_features(epochs)
