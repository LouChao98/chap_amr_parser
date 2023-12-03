import penman

for graph in penman.iterdecode(open("convert_in.txt")):
    print(str(graph.triples).replace("'", ""))
