import csv


def retrieve_all_data(folders):
    epochs=[int(e.split("epoch")[1]) for e in folders]
    stat={e:{} for e in epochs}
    for folder in folders:
        file = folder+"/rel_full.csv"
        with open(file) as relevant:
            reader = csv.DictReader(relevant)
            for row in reader:
                if row['username']!='nimo':
                    key = row['orig__id']
                    epoch = int(folder.split("epoch")[1])
                    if row['this_document_is'].lower()=='relevant':
                        stat[epoch][key]=stat[epoch].get(key,0)+1
                    else:
                        stat[epoch][key] = stat[epoch].get(key, 0)
    return stat


def get_mapping(mapping_file):
    mapping = {}
    with open(mapping_file) as mapping_data:
        for data in mapping_data:
            splitted = data.split()
            name = splitted[2]
            if not name.__contains__("originalDoc"):
                id = name.split("-")[1]
                new_id = "ObjectId("+id+")"
                mapping[new_id] = splitted[0]
    return mapping


def write_relevance_file(stat,mapping):
    file = open("qrel_asr",'w')
    last_rel ={}
    docs = list(stat[1].keys())
    epochs = sorted(list(stat.keys()))
    for doc in docs:
        for e in epochs:
            if stat[e].get(doc,False):

                last_rel[doc] = stat[e][doc]
            doc_name = "ROUND-"+str(e).zfill(2)+"-"+mapping[doc]+"-"+doc
            if last_rel[doc]==5:
                line = mapping[doc]+" 0 "+doc_name+" "+str(1)+"\n"
            else:
                line = mapping[doc]+" 0 "+doc_name+" "+str(0)+"\n"
            file.write(line)
    file.close()

def new_rel(rel):
    f=open("new_rel",'w')
    with open(rel) as r:
        for i in r:
            splitted=i.split()
            if splitted[3].rstrip()=="3":
                line = splitted[0]+" "+splitted[1]+" "+splitted[2]+" 1\n"
            else:
                line = splitted[0] + " " + splitted[1] + " " + splitted[2] + " 0\n"
            f.write(line)
    f.close()

folders = ["C:\\study\\thesis - new\\coampetition\\epoch1","C:\\study\\thesis - new\\competition\\epoch2","C:\\study\\thesis - new\\competition\\epoch3","C:\\study\\thesis - new\\competition\\epoch4"]
mapping_file = "C:\\study\\thesis - new\\competition\\mapping"


stat=retrieve_all_data(folders)
mapping = get_mapping(mapping_file)
write_relevance_file(stat, mapping)
# new_rel("documents_updated.rel")