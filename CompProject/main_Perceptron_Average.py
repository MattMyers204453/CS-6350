from asyncore import read
import pandas as p
import os
import sys
import numpy as np
import read_data_svm as read
from scipy import optimize
from scipy.optimize import Bounds
import math

TRAIN_PATH = "train_final.csv"
TEST_PATH = "test_final.csv"

TRAIN_SIZE = 23844
TEST_SIZE = 1000000

def read_data_into_dataframe_Perceptron(file_name, attributes, test, max):
    dict = {}
    for i in range(len(attributes)):
        dict[attributes[i]] = []
    count = 0
    with open(os.path.join(sys.path[0], file_name)) as file:
        for line in file:
            if count == 0:
                count += 1
                continue
            if count >= max + 1:
                break
            terms = line.strip().split(",")
            if test:
                terms = terms[1:]
            for i in range(len(attributes)):
                dict[attributes[i]].append(terms[i]) #= terms[i]
            count += 1
    df = p.DataFrame(dict)
    return df

def preprocess_dataframe_Perceptron(df, attributes, attribute_values, test):
    for a in attributes:
        df.loc[df[a] == '?', a] = df[a].mode()[0]
    standardize_col(df, "age")
    one_hot_encode(df, "workclass", attribute_values)
    df.drop("fnlwgt", inplace=True, axis=1)
    one_hot_encode(df, "education", attribute_values)
    standardize_col(df, "education.num")
    one_hot_encode(df, "marital.status", attribute_values)
    one_hot_encode(df, "occupation", attribute_values)
    one_hot_encode(df, "relationship", attribute_values)
    one_hot_encode(df, "race", attribute_values)
    one_hot_encode(df, "sex", attribute_values)
    standardize_col(df, "capital.gain")
    standardize_col(df, "capital.loss")
    standardize_col(df, "hours.per.week")
    one_hot_encode(df, "native.country", attribute_values)
    if not test:
        map = {1: 1, 0: -1}
        df["income>50K"] = df["income>50K"].astype("int").map(map)
        label_column_to_move = df.pop("income>50K")
        df.insert(len(df.columns), "income>50K", label_column_to_move)


def standardize_col(df, col_name):
    df[col_name] = df[col_name].astype(float)
    df[col_name] = (df[col_name] - df[col_name].mean()) / df[col_name].std()

def one_hot_encode(df, col_name, attribute_values):
    new_categories = attribute_values[col_name]
    for new_category in new_categories:
        df[new_category] = [0] * len(df.index)
    for i in range(len(df.index)):
        value_with_1 = df.at[i, col_name]
        for new_category in new_categories:
            df.at[i, new_category] = 1.0 if value_with_1 == new_category else -1.0
    df.drop(col_name, inplace=True, axis=1)
    


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

attributes = ["age","workclass","fnlwgt","education","education.num","marital.status","occupation", "relationship","race", "sex","capital.gain","capital.loss","hours.per.week","native.country","income>50K"]
# df = read_data_into_dataframe_NN(TRAIN_PATH, attributes, False, 150)
# preprocess_dataframe_NN(df, attributes, attribute_values, False)
# #sys.displayhook(df)

# Perceptron
##############################################
def get_x_vector_at(i, df):
    row = df.iloc[i]
    row_as_list = row.to_list()[:-1]
    row_as_list.append(1)
    row_as_matrix = np.array(row_as_list, dtype=float)
    return row_as_matrix

def get_x_vector_at_DATA_HAS_NO_LABEL(i, df):
    row = df.iloc[i]
    row_as_list = row.to_list()
    row_as_list.append(1)
    row_as_matrix = np.array(row_as_list, dtype=float)
    return row_as_matrix

def sgn(v):
    if v > 0:
        return 1
    return -1

def update(w, y, x_vector, r):
    w_new = np.add(w, r * y * x_vector)
    return w_new

def predict(w, x_vector):
    return sgn(np.dot(w, x_vector))

##############################################################################################################################

#df = p.read_pickle(f"df_{TRAIN_SIZE}.pkl")

print("Reading train data...")
df = read_data_into_dataframe_Perceptron(TRAIN_PATH, attributes, False, TRAIN_SIZE)
print("Preprocessing train data...")
preprocess_dataframe_Perceptron(df, attributes, attribute_values, False)
sys.displayhook(df)

df.to_pickle(f"df_{TRAIN_SIZE}.pkl")
print(f"SAVED DF_{TRAIN_SIZE}")

#------

num_of_features = len(df.columns) - 1 # minus two to drop off the label and fnlwgt
#w_as_list = [0] * (num_of_features + 1) # plus one to account for constant in w-vector (notational sugar)
w_as_list = [0] * (len(df.columns) - 1 + 1) # plus one to account for constant in w-vector (notational sugar)
w = np.array(w_as_list)
m = len(df.index)

a = [0] * (len(w_as_list))
T = int(sys.argv[1]) if len(sys.argv) == 3 else 10
r = float(sys.argv[2]) if len(sys.argv) == 3 else 1.0
print(f"Epochs: {T}")
print(f"Learning rate = {r}")
print("Training model...")
for j in range(T):
    df = df.sample(frac=1)
    for i in range(len(df.index)):
        row = df.iloc[i]
        x_vector = get_x_vector_at(i, df)
        actual_value = float(row.get("income>50K"))
        prediction = predict(w, x_vector)
        if prediction != actual_value:
            w = update(w, actual_value, x_vector, r)
        a = np.add(a, w)

print("Testing model on training data...")
errors = 0
for i in range(len(df.index)):
    row = df.iloc[i]
    x_vector = get_x_vector_at(i, df)
    actual = float(row.get("income>50K"))
    guess = predict(a, x_vector) 
    if guess != actual:
        errors += 1

print(f"TOTAL MISCLASSIFIED: {errors}")
print(f"---TRAIN ACCURACY: {((float(TRAIN_SIZE) - errors) / float(TRAIN_SIZE)) * 100}%")

# print("Reading test data...")
# test_df = read_data_into_dataframe_Perceptron(TEST_PATH, attributes[:-1], True, TEST_SIZE)
# print("Preprocessing test data...")
# preprocess_dataframe_Perceptron(test_df, attributes[:-1], attribute_values, True)

# test_df.to_pickle(f"test_df_{TEST_SIZE}.pkl")
# print("DONE")

test_df = p.read_pickle("test_df_{TEST_SIZE}.pkl")
sys.displayhook(test_df)

predictions = [0] * len(test_df.index)
print("Making predictions on test data...")
error_count = 0
for i in range(len(test_df.index)):
    x_vector = get_x_vector_at_DATA_HAS_NO_LABEL(i, test_df)
    prediction = predict(a, x_vector)
    predictions[i] = prediction

print("Writing predictions to file...")
#test = p.read_csv(os.path.join(sys.path[0], 'test_final.csv'))
ID_s = [1] * len(test_df)
for i in range(len(test_df)):
    ID_s[i] = i + 1
submission = p.DataFrame({'Id': ID_s, 'Prediction': predictions})
submission.to_csv("submission.csv", index=False)
#------

