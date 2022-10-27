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
    return p.DataFrame(dict)