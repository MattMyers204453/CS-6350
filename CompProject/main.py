from asyncore import read
import pandas as p
import os
import sys
import random

import ID3 as id3


# # Read the data
# train = pd.read_csv('../input/train.csv')

# # pull data into target (y) and predictors (X)
# train_y = train.SalePrice
# predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

# # Create training predictors data
# train_X = train[predictor_cols]

# my_model = RandomForestRegressor()
# my_model.fit(train_X, train_y)

# test = pd.read_csv(os.path.join(sys.path[0], 'test_final.csv'))
# #label_col = train["income>50K"]

# predictions = [1] * 23842
# for i in range(23842):
#     rand = random.choice([0, 1])
#     predictions[i] = rand

# submission = pd.DataFrame({'Id': test.ID, 'Prediction': predictions})

# submission.to_csv("submission.csv", index=False)


def read_data_into_dataframe(file_name, attributes, continuous_attributes, test, max):
    dict = {}
    for i in range(len(attributes)):
        dict[attributes[i]] = []
    count = 0
    with open(os.path.join(sys.path[0], file_name)) as file:
        for line in file:
            if count == 0:
                count += 1
                continue
            if count > max:
                break
            terms = line.strip().split(",")
            if test:
                terms = terms[1:]
            for i in range(len(attributes)):
                dict[attributes[i]].append(terms[i]) #= terms[i]
            count += 1
    df = p.DataFrame(dict)
    for a in attributes:
        df.loc[df[a] == '?', a] = df[a].mode()[0]
    for ca in continuous_attributes:
        df[ca] = df[ca].astype(int).gt(df[ca].median()).astype(int)
    return df

attribute_values = {}
attribute_values["age"] = [0, 1]
attribute_values["workclass"] = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
attribute_values["fnlwgt"] = [0, 1]
attribute_values["education"] = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
attribute_values["education.num"] = [0, 1]
attribute_values["marital.status"] = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
attribute_values["occupation"] = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
attribute_values["relationship"] = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
attribute_values["race"] = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
attribute_values["sex"] = ["Female", "Male"]
attribute_values["capital.gain"] = [0, 1]
attribute_values["capital.loss"] = [0, 1]
attribute_values["hours.per.week"] = [0, 1]
attribute_values["native.country"] = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
attribute_values["income>50K"] = [0, 1]

labels = [0, 1]

attributes = ["age","workclass","fnlwgt","education","education.num","marital.status","occupation", "relationship","race", "sex","capital.gain","capital.loss","hours.per.week","native.country","income>50K"]
categorical_attributes = ["workclass","education","marital.status","occupation", "relationship","race", "sex","native.country"]
continuous_attributes = ["age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week"]
df = read_data_into_dataframe("train_final.csv", attributes, continuous_attributes, False, 10000)
#sys.displayhook(df)
test_df = read_data_into_dataframe("test_final.csv", attributes[:-1], continuous_attributes, True, 1000000)
#sys.displayhook(test_df)

predictions = [0] * len(test_df)
root = id3.ID3(df, attributes, attribute_values, labels)
error_count = 0
for i in range(len(test_df.index)):
    row = test_df.iloc[i]
    result_label = id3.traverse_tree(row, root, attribute_values)
    predictions[i] = result_label

#test = p.read_csv(os.path.join(sys.path[0], 'test_final.csv'))
ID_s = [1] * len(test_df)
for i in range(len(test_df)):
    ID_s[i] = i + 1
submission = p.DataFrame({'Id': ID_s, 'Prediction': predictions})
submission.to_csv("submission.csv", index=False)