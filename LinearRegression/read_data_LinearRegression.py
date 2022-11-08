import os
import sys
import pandas as p

def read_data_into_dataframe(file_name, attributes, how_many):
    dict = {}
    for i in range(len(attributes)):
        dict[attributes[i]] = []
    count = 0
    with open(os.path.join(sys.path[0], file_name)) as file:
        for line in file:
            if count >= how_many:
                break
            terms = line.strip().split(",")
            for i in range(len(attributes)):
                dict[attributes[i]].append(terms[i]) 
            count += 1
    df = p.DataFrame(dict)
    df['Cement'] = df['Cement'].astype('double')
    df['Slag'] = df['Slag'].astype('double')
    df['Fly ash'] = df['Fly ash'].astype('double')
    df['Water'] = df['Water'].astype('double')
    df['SP'] = df['SP'].astype('double')
    df['Coarse Aggr'] = df['Coarse Aggr'].astype('double')
    df['Fine Aggr'] = df['Fine Aggr'].astype('double')
    df['SLUMP'] = df['SLUMP'].astype('double')
    return df