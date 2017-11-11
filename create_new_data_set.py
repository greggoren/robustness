def read_original_data_set(data_set_file):
    data_records = {}
    queries ={}
    with open(data_set_file) as data_set:
        for data in data_set:
            name = data.split(" # ")[1].rstrip()
            queries[name]=[]
            query = data.split()[1].replace("qid:","")
            if not data_records.get(query,False):
                data_records[query]={}
            data_records[query][name] = data.split(" # ")[0]
    for query in data_records:
        for doc in data_records[query]:
            queries[doc].append(query)
    return data_records,queries

def retrieve_spam_score(spam_file,queries):
    minimum,maximum={},{}
    scores_tmp={}
    with open(spam_file) as spam_scores:
        for spam_score in spam_scores:
            score,doc=spam_score.split()[0],spam_score.split()[1].rstrip()
            score = int(score)
            if not queries.get(doc,False):
                continue
            scores_tmp[doc]=score
            for query in queries[doc]:
                if score>=maximum.get(query,score):
                    maximum[query]=score
                elif score<=minimum.get(query,score):
                    minimum[query] = score
    scores={}
    for doc in scores_tmp:
        scores[doc]={}
        for query in queries[doc]:
            scores[doc][query]=float(scores_tmp[doc]-minimum[query])/(maximum[query]-minimum[query])
    return scores

def create_data_set(data_records,scores):
    with open("ClueWeb09Extra",'w') as file:
        for query in data_records:
            for doc in data_records[query]:
                line = data_records[query][doc]+" 26:"+str(scores[doc][query])+" # "+doc+"\n"
                file.write(line)


data_records,queries=read_original_data_set("data/featuresCB_asr")
scores=retrieve_spam_score("debug",queries)
#"clueweb09spam.Fusion"
print("scores=",scores)
print("queries",queries)
create_data_set(data_records,scores)