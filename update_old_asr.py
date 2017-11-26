import csv
import xml.etree.ElementTree as ET

def retrieve_initial_documents():
    initial_query_docs={}
    tree = ET.parse('reports/documents.trectext')
    root = tree.getroot()
    for doc in root:
        name =""
        for att in doc:
            if att.tag == "DOCNO":
                id = att.text
                if id.split("-")[1]=="00":
                    break
                else:name=id
            else:
                initial_query_docs[name]=att.text.rstrip()
    return initial_query_docs,initial_query_docs.keys()






def get_report(d):
    map = {}
    mapped=[]
    for i in range(4,9):
        with open("reports/"+str(i)+".csv",encoding='utf8') as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                if row['user_name']=="nimo":
                    continue
                content=row['post_content'].replace('&','and').rstrip().replace("\n","").replace(" ","")
                query = row['query_num'].zfill(3)
                epoch = i
                for id in d:
                    if epoch==int(id.split("-")[1]) and query==id.split("-")[2]:
                        doc = d[id].rstrip().replace("\n","").replace(" ","")
                        if doc == content and id.split("-")[3] not in mapped:
                            map[row['user_name']]=id.split("-")[3]
                            mapped.append(id.split("-")[3])

    return map




def retrieve_stats_from_cf(map):
    spam_ks={}

    for i in range(1,9):
        with open('reports/'+str(i)+'.csv',encoding='utf8') as csv_file:
            spam_ks[i]={}
            reader = csv.DictReader(csv_file)
            for row in reader:
                if not spam_ks[i].get(row['query_num'].zfill(3),False) and row['user_name']!='nimo':
                    spam_ks[i][row['query_num'].zfill(3)]={}
                if row['user_name']!='nimo' and (row.get('this_document_is',"VALID")!="VALID" or row.get('check_one',"VALID")!="VALID"):
                    spam_ks[i][row['query_num'].zfill(3)][map[row['user_name']]]=spam_ks[i][row['query_num'].zfill(3)].get(map[row['user_name']],0)+1
                elif row['user_name']!='nimo':
                    spam_ks[i][row['query_num'].zfill(3)][map[row['user_name']]]=spam_ks[i][row['query_num'].zfill(3)].get(map[row['user_name']],0)
    queries = set(spam_ks[1].keys())
    for epoch in spam_ks:

        for query in queries:
            if epoch==1:
                continue
            old = set(spam_ks[epoch-1][query].keys())
            if not spam_ks[epoch].get(query,False):
                spam_ks[epoch][query]=spam_ks[epoch-1][query]
            new = set(spam_ks[epoch][query].keys())
            missing_docs=old.difference(new)
            for doc in missing_docs:
                spam_ks[epoch][query][doc]=spam_ks[epoch-1][query][doc]
    return spam_ks

def stats(spam_ks):
    maximum,minimum={},{}
    scores_tmp = {}
    for epoch in spam_ks:
        scores_tmp[epoch]={}
        maximum[epoch]={}
        minimum[epoch]={}
        for query in spam_ks[epoch]:
            scores_tmp[epoch][query]={}
            for doc in spam_ks[epoch][query]:
                scores_tmp[epoch][query][doc]=100-20*spam_ks[epoch][query][doc]
                if scores_tmp[epoch][query][doc] >= maximum[epoch].get(query,scores_tmp[epoch][query][doc]):
                    maximum[epoch][query]=scores_tmp[epoch][query][doc]
                if scores_tmp[epoch][query][doc] <= minimum[epoch].get(query,scores_tmp[epoch][query][doc]):
                    minimum[epoch][query]=scores_tmp[epoch][query][doc]

    scores={}
    for epoch in scores_tmp:
        scores[epoch]={}
        for query in scores_tmp[epoch]:
            scores[epoch][query]={}
            for doc in scores_tmp[epoch][query]:
                if maximum[epoch][query]!=minimum[epoch][query]:
                    scores[epoch][query][doc]=float(scores_tmp[epoch][query][doc] - minimum[epoch][query])/(maximum[epoch][query]-minimum[epoch][query])
                else:
                    scores[epoch][query][doc]=0
    return scores

def rewrite_file(features,scores):
    new_asr = open("features_asr_modified",'w')
    with open(features) as data_set:
        for data in data_set:
            name = data.split(" # ")[1]
            if name.split("-")[1]=="00":
                continue
            epoch = int(name.split("-")[1])
            query = name.split("-")[2]
            doc = name.split("-")[3].rstrip()
            data_rec = data.split(" # ")[0]
            try:
                new_line = data_rec+" 26:"+str(scores[epoch][query][doc])+" # "+name
            except:
                print(epoch,query,doc)
            new_asr.write(new_line)
    new_asr.close()

d,_=retrieve_initial_documents()
map=get_report(d)
spam_ks=retrieve_stats_from_cf(map)
spam_ks[1]['010']['09']=0
spam_ks[1]['010']['49']=0
spam_ks[1]['011']['09']=0
spam_ks[1]['033']['12']=0
spam_ks[1]['051']['51']=0
spam_ks[1]['051']['07']=0
spam_ks[1]['098']['04']=0
spam_ks[1]['144']['09']=0
spam_ks[1]['164']['21']=0
spam_ks[1]['164']['51']=0
spam_ks[1]['166']['12']=0
spam_ks[1]['167']['21']=0
spam_ks[1]['177']['21']=0
spam_ks[1]['195']['51']=0
scores=stats(spam_ks)
rewrite_file("data/features_asr",scores)