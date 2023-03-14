import sys
import re

def convert_coref(annot):
    if annot == "-":
        return "_"
    is_open = False
    parts = [p for p in re.split(r'([()])', annot) if p != ""]
    new_parts = []
    for part in parts:
        if part == "(":
            is_open = True
        elif part != ")" and is_open:
            part = "e" + part + "-x-1-"
        elif part != ")" and not is_open:
            part = "e" + part
        new_parts.append(part)
    return "Entity="+"".join(new_parts) 

docname = None
sent_i = 1
i = 1
header = []
for line in sys.stdin:
    line = line.rstrip()
    if line.startswith("#begin"):
        m = re.match(r"#begin document \((.*)\);", line)
        docname = re.sub(r'/', '.', m.group(1))
        header.append("# newdoc id = {:s}".format(docname))
        header.append("# global.Entity = eid-etype-head-other")
        header.append("# sent_id = {:s}-{:d}".format(docname, sent_i))
        header.append("# text = sentence")
    elif line.startswith("#end"):
        if not header:
            print()
        continue
    elif line == "":
        print("")
        i = 1
        sent_i += 1
        header.append("# sent_id = {:s}-{:d}".format(docname, sent_i))
        header.append("# text = sentence")
    else:
        if header:
            for hline in header:
                print(hline)
            header = []
        cols = re.split("\s+", line)
        coref = convert_coref(cols[-1])
        # if len(cols) == 2:
        feats = [str(i)] + ["_"]*5 + ["0"] + ["_"]*2 + [coref]
        print("\t".join(feats))
        i += 1
