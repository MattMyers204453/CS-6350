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
                dict[attributes[i]].append(terms[i]) #= terms[i]
            count += 1
    return p.DataFrame(dict)

def convert_dataframe(df):
    df['age'] = df['age'].astype('int')
    df['balance'] = df['balance'].astype('int')
    df['day'] = df['day'].astype('int')
    df['duration'] = df['duration'].astype('int')
    df['campaign'] = df['campaign'].astype('int')
    df['pdays'] = df['pdays'].astype('int')
    df['previous'] = df['previous'].astype('int')

    age_median = df['age'].median()
    balance_median = df['balance'].median()
    day_median = df['day'].median()
    duration_median = df['duration'].median()
    campaign_median = df['campaign'].median()
    pdays_median = df['pdays'].median()
    previous_median = df['previous'].median()

    df['age'] = df['age'].gt(age_median).astype(int)
    df['balance'] = df['balance'].gt(balance_median).astype(int)
    df['day'] = df['day'].gt(day_median).astype(int)
    df['duration'] = df['duration'].gt(duration_median).astype(int)
    df['campaign'] = df['campaign'].gt(campaign_median).astype(int)
    df['pdays'] = df['pdays'].gt(pdays_median).astype(int)
    df['previous'] = df['previous'].gt(previous_median).astype(int)

    binary = {'no': 0, 'yes': 1}
    label_binary = {'no': -1, 'yes': 1}

    df["default"] = df["default"].map(binary).astype(int)
    df["loan"] = df["loan"].map(binary).astype(int)
    df["housing"] = df["housing"].map(binary).astype(int)
    df["y"] = df["y"].map(label_binary).astype(int)
    return df