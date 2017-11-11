def read_original_data_set(data_set_file):
    data_records = {}
    queries ={}
    with open(data_set_file) as data_set:
        for data in data_set:
            name = data.split(" # ")[1].rstrip()
            query = data.split()[1].split("qid:")[0]
            data_records[name] = data.split(" # ")[0]
            queries[name]=query
    return data_records,queries

def retrieve_spam_score(spam_file,queries):
    minimum,maximum={},{}
    scores_tmp={}
    with open(spam_file) as spam_scores:
        for spam_score in spam_scores:
            score,doc=spam_score.split()[0],spam_score.split()[1].rstrip()
            score = int(score)
            if queries.get(doc,False):
                scores_tmp[doc]=score
                if score>=maximum.get(queries[doc],score):
                    maximum[queries[doc]]=score
                elif score<=minimum.get(queries[doc],score):
                    minimum[queries[doc]] = score
    scores={}
    for doc in scores_tmp:
        query = queries[doc]
        scores[doc]=float(scores_tmp[doc]-minimum[query])/(maximum[query]-minimum[query])
    return scores

def create_data_set(data_records,scores):
    with open("ClueWeb09Extra",'w') as file:
        for doc in data_records:
            line = data_records[doc]+" 26:"+str(scores[doc])+" # "+doc+"\n"
            file.write(line)


data_records,queries=read_original_data_set("featuresCB_asr")
scores=retrieve_spam_score("clueweb09spam.Fusion",queries)
create_data_set(data_records,scores)