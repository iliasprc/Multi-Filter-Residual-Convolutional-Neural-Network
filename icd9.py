'/mnt/784C5F3A4C5EF1FC/PROJECTS/XAI_research/data/ICD9CM.ttl'
#from rdflib import Graph
import csv
import pandas as pd
import numpy as np
def reaf_ttl():
    g = Graph()
    g.parse("./../../data/ICD9CM.ttl",format='turtle')



    #
    g = Graph()
    g.parse('/mnt/784C5F3A4C5EF1FC/PROJECTS/XAI_research/data/owlapi.xrdf')
    # print(len(g))
    for subj, pred, obj in g:

        print((subj, pred, obj))
        # check if there is at least one triple in the Graph
        if (subj, pred, obj) not in g:
           raise Exception("Not in graph")

    print(len(g))
    # import pprint
    # for stmt in g:
    #     pprint.pprint(stmt)

def read_icd9_ontology_csv(path):
    data = pd.read_csv(path)
    rows, columns = data.shape
    print(rows,columns)
    adjacency = np.zeros((rows,rows)).astype(np.uint8)
    icd9_to_index = {}
    # print(data)
    print(data.columns)
    # print(data.dtypes)
    # print('\n')
    print(data.loc[0,:])
    # print('\n')
    # print(data.loc[1, :])
    # print('\n')
    # print(data.loc[0,'Class ID'])
    for i in range(rows):
        class_icd9_code = data.loc[i,'Class ID']
        icd9_to_index[class_icd9_code] = i
        #print(class_icd9_code)

    #
    for i in range(rows):
        #print(data.loc[i, :])
        class_icd9_code = data.loc[i,'Class ID']
        class_index = icd9_to_index[class_icd9_code]
        parent_id = data.loc[i, 'Parents']
        if parent_id not in icd9_to_index:
            print(f'  {i} REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEe \n\n\n\n')
        else:
            parent_index = icd9_to_index[parent_id]
            #print(f'Iteration {i}/{rows} Parent {parent_index} Child {class_index} ')
            adjacency[parent_index,class_index] = 1
    np.save("./data/adjacency.npy", adjacency)
    import scipy.io as sio
    sio.savemat("./data/adjacency.mat", {"graph_sparse": adjacency})
    f = open("./icd9cm/icd9_to_index.txt","w")
    f.write( str(icd9_to_index) )
    f.close()
    # import json
    # json = json.dumps(icd9_to_index)
    # f = open("./icd9cm/icd9_to_index.json","w")
    # f.write(json)
    # f.close()


read_icd9_ontology_csv('./data/ICD9CM.csv')



