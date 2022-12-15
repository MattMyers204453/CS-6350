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

    df['variance'] = df['variance'].astype('double')
    df['skewness'] = df['skewness'].astype('double')
    df['kurtosis'] = df['kurtosis'].astype('double')
    df['entropy'] = df['entropy'].astype('double')
    df['y'] = df['y'].astype('int')
    #map = {1: 1, 0: -1}
    #df["y"] = df["y"].map(map)
    return df