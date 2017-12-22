f = open("new_rel_updated", 'w')
with open("documents_updated.rel") as relevance:
    for rel in relevance:

        splitted = rel.split()
        if splitted[-1].rstrip() == "2":
            last = "1"
        elif splitted[-1].rstrip() == "3":
            last = "2"
        else:
            last = "0"
        line = " ".join(list(splitted[:-1])) + " " + last + "\n"
        f.write(line)
f.close()
