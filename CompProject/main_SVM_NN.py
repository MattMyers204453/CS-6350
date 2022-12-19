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

TRAIN_SIZE = 3000
TEST_SIZE = 1000000

def read_data_into_dataframe_NN(file_name, attributes, test, max):
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

def preprocess_dataframe_NN(df, attributes, attribute_values, test):
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
    # if not test:
    #     map = {1: 1, 0: -1}
    #     df["income>50K"] = df["income>50K"].astype("int").map(map)


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

# Neural Network
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

def round(v):
    if v >= 0.5:
        return 1
    return 0

ONE = 1
TWO = 2
THREE = 3
numFeatures = len(attributes) - 2 # Dropped fnlwgt and income>50K
num_of_zs_at_layer_2 = 20
num_of_zs_at_layer_1 = 20
num_of_xs_at_layer_0 = numFeatures + 1
# num_of_zs_at_layer_2 = 3
# num_of_zs_at_layer_1 = 3
# num_of_xs_at_layer_0 = 3
weights_dict = {}
z_dict = {}

def initialize_weights():
    # layer 3
    for i in range(num_of_zs_at_layer_2):
        weights_dict[w(i, ONE, THREE)] = np.random.normal(0, 1)
    # layer 2
    for n in range(1, num_of_zs_at_layer_2):
        for m in range(num_of_zs_at_layer_1):
            weights_dict[w(m, n, TWO)] = np.random.normal(0, 1)
    # layer 1
    for n in range(1, num_of_zs_at_layer_1):
        for m in range(num_of_xs_at_layer_0):
            weights_dict[w(m, n, ONE)] = np.random.normal(0, 1)

def w(start, to, layer):
    return f"w_{start}{to}^{layer}"

def z(num, layer):
    return f"z_{num}^{layer}"

def loss_partial(y, y_star):
    return y - y_star

def sigmoid(x):
    if (x > 24):
        return 1
    if (x < -24):
        return 0
    return 1.0 / (1.0 + math.exp(-1 * x))

def sigmoid_partial(s):
    return sigmoid(s) * (1 - sigmoid(s))

def get_s_for_given_layer2_z(z_num):
    s = 0
    for i in range(num_of_zs_at_layer_1):
        s += (weights_dict[w(i, z_num, TWO)] * z_dict[z(i, ONE)])
    return s

def get_s_for_given_layer1_z(z_num, x_input):
    s = 0.0
    for i in range(num_of_xs_at_layer_0):
        s += (weights_dict[w(i, z_num, ONE)] * x_input[i])
    return s


def get_layer_3_partials(y, y_star):
    L_partial = loss_partial(y, y_star)
    grad_weights_layer_3 = {}
    for m in range(num_of_zs_at_layer_2):
        grad_weights_layer_3[w(m, ONE, THREE)] =  L_partial * z_dict[z(m, 2)]
    return grad_weights_layer_3

def get_layer_2_partials(y, y_star):
    L_partial = loss_partial(y, y_star)
    grad_weights_layer_2 = {}
    for n in range(1, num_of_zs_at_layer_2):
        s = get_s_for_given_layer2_z(n)
        sig_partial = sigmoid_partial(s)
        for m in range(num_of_zs_at_layer_1):
            grad_weights_layer_2[w(m, n, TWO)] = L_partial * weights_dict[w(n, ONE, THREE)] * sig_partial * z_dict[z(m, ONE)]
    return grad_weights_layer_2

def get_layer_1_partials(y, y_star, x_input):
    L_partial = loss_partial(y, y_star)
    grad_weights_layer_1 = {}
    for n in range(1, num_of_zs_at_layer_1):
        path_sum = 0
        for p in range(1, num_of_zs_at_layer_2):
            s = get_s_for_given_layer2_z(p)
            sig_partial = sigmoid_partial(s)
            path_sum += (L_partial * weights_dict[w(p, ONE, THREE)] * sig_partial * weights_dict[w(n, p, TWO)])
        outside_s = get_s_for_given_layer1_z(n, x_input)
        term_outside_bracket = sigmoid_partial(outside_s)
        for m in range(len(x_input)):
            grad_weights_layer_1[w(m, n, ONE)] = path_sum * term_outside_bracket * x_input[0]
    return grad_weights_layer_1

def forward_pass(x_input):
    # get first layer z's
    z_dict[z(0, ONE)] = 1
    for i in range(1, num_of_zs_at_layer_1):
        s = get_s_for_given_layer1_z(i, x_input)
        sig = sigmoid(s)
        z_dict[z(i, ONE)] = sig
    # get second layer z's
    z_dict[z(0, TWO)] = 1
    for i in range(1, num_of_zs_at_layer_2):
        s= get_s_for_given_layer2_z(i)
        sig = sigmoid(s)
        z_dict[z(i, TWO)] = sig
    # get y 
    y_output = 0
    for i in range(num_of_zs_at_layer_2):
        y_output += (weights_dict[w(i, ONE, THREE)] * z_dict[z(i, TWO)])
    return y_output

def back_propagation(y, y_star, x_input):
    w_layer_3_partials = get_layer_3_partials(y, y_star)
    w_layer_2_partials = get_layer_2_partials(y, y_star)
    w_layer_1_partials = get_layer_1_partials(y, y_star, x_input)
    return (w_layer_1_partials, w_layer_2_partials, w_layer_3_partials)

def update_weights(r, w_partials_tuple):
    w_layer_3_partials = w_partials_tuple[2]
    w_layer_2_partials = w_partials_tuple[1]
    w_layer_1_partials = w_partials_tuple[0]
    # layer 3
    for i in range(num_of_zs_at_layer_2):
        weights_dict[w(i, ONE, THREE)] -= r * w_layer_3_partials[w(i, ONE, THREE)]
    # layer 2
    for n in range(1, num_of_zs_at_layer_2):
        for m in range(num_of_zs_at_layer_1):
            weights_dict[w(m, n, TWO)] -= r * w_layer_2_partials[w(m, n, TWO)]
    # layer 1
    for n in range(1, num_of_zs_at_layer_1):
        for m in range(num_of_xs_at_layer_0):
            weights_dict[w(m, n, ONE)] -= r * w_layer_1_partials[w(m, n, ONE)]

def norm_gradient(w_partials_tuple):
    w_layer_3_partial_values = list(w_partials_tuple[2].values())
    w_layer_2_partials_values = list(w_partials_tuple[1].values())
    w_layer_1_partials_values = list(w_partials_tuple[0].values())
    w_partial_values = np.asarray(w_layer_3_partial_values + w_layer_2_partials_values + w_layer_1_partials_values)
    return np.linalg.norm(w_partial_values)


##############################################################################################################################

# df = p.read_pickle(f"df_{TRAIN_SIZE}.pkl")

print("Reading train data...")
df = read_data_into_dataframe_NN(TRAIN_PATH, attributes, False, TRAIN_SIZE)
print("Preprocessing train data...")
preprocess_dataframe_NN(df, attributes, attribute_values, False)
sys.displayhook(df)

df.to_pickle(f"df_{TRAIN_SIZE}.pkl")
print(f"SAVED DF_{TRAIN_SIZE}")

#------
initialize_weights()

T = 20
r = 0.5
d= 0.3
print(f"Epochs: {T}")
print(f"Learning rate = {r}")
print(f"Width: {num_of_zs_at_layer_1}")
print("Training model...")
# last_gradient_norm = 10000
# stop_early = False
for t in range(T):
    df = df.sample(frac=1)
    r = r / float(1 + (r * t) / float(d))
    for i in range(len(df.index)):
        row = df.iloc[i]
        x_i = get_x_vector_at(i, df)
        x_input = np.insert(x_i, 0, 1)[:-1]
        y_star = float(row.get("income>50K"))
        prediction = forward_pass(x_input)
        if y_star != round(prediction):
            gradient_of_loss = back_propagation(y=prediction, y_star=y_star, x_input=x_input)
            update_weights(r, gradient_of_loss)

print("Testing model on training data...")
errors = 0
for i in range(TRAIN_SIZE):
    row = df.iloc[i]
    x_vector = get_x_vector_at(i, df)
    x_input = np.insert(x_vector, 0, 1)[:-1]
    actual = float(row.get("income>50K"))
    prediction = round(forward_pass(x_input)) 
    if actual != prediction:
        errors += 1

print(f"TOTAL MISCLASSIFIED: {errors}")
print(f"---TRAIN ACCURACY: {((float(TRAIN_SIZE) - errors) / float(TRAIN_SIZE)) * 100}%")

# print("Reading test data...")
# test_df = read_data_into_dataframe_NN(TEST_PATH, attributes[:-1], True, TEST_SIZE)
# print("Preprocessing test data...")
# preprocess_dataframe_NN(test_df, attributes[:-1], attribute_values, True)

# test_df.to_pickle("test_df.pkl")
# print("DONE")

test_df = p.read_pickle("test_df.pkl")
sys.displayhook(test_df)

predictions = [0] * len(test_df)
print("Making predictions on test data...")
error_count = 0
for i in range(len(test_df.index)):
    x_i = get_x_vector_at_DATA_HAS_NO_LABEL(i, test_df)
    x_input = np.insert(x_i, 0, 1)#[:-1]
    prediction = round(forward_pass(x_input))
    predictions[i] = prediction

print("Writing predictions to file...")
#test = p.read_csv(os.path.join(sys.path[0], 'test_final.csv'))
ID_s = [1] * len(test_df)
for i in range(len(test_df)):
    ID_s[i] = i + 1
submission = p.DataFrame({'Id': ID_s, 'Prediction': predictions})
submission.to_csv("submission.csv", index=False)
#------

# print("Testing model on training data...")
# errors = 0
# for i in range(len(df.index)):
#     row = df.iloc[i]
#     x_vector = get_x_vector_at(i, df)
#     actual = row.get("income>50K")
#     guess = predict(w, x_vector) 
#     if guess != actual:
#         errors += 1

# print(f"TOTAL MISCLASSIFIED: {errors}")
# print(f"---TRAIN ACCURACY: {((float(len(df.index)) - errors) / float(len(df.index))) * 100}%")
# #print(f"***MODEL PARAMETERS: {w}")
# ##################


# #test_df = read_data_into_dataframe_ID3("test_final.csv", attributes[:-1], continuous_attributes, True, 100)
# #sys.displayhook(test_df)

# print("Reading test data...")
# test_df = read_data_into_dataframe_NN(TEST_PATH, attributes[:-1], True, 10)
# print("Preprocessing test data...")
# preprocess_dataframe_NN(test_df, attributes[:-1], attribute_values, True)
# predictions = [0] * len(test_df)

# print("Testing model on test data...")
# error_count = 0
# for i in range(len(test_df.index)):
#     row = test_df.iloc[i]
#     x_vector = get_x_vector_at_DATA_HAS_NO_LABEL(i, test_df)
#     guess = predict(w, x_vector)
#     predictions[i] = guess

# print("Writing predictions to file...")
# #test = p.read_csv(os.path.join(sys.path[0], 'test_final.csv'))
# ID_s = [1] * len(test_df)
# for i in range(len(test_df)):
#     ID_s[i] = i + 1
# submission = p.DataFrame({'Id': ID_s, 'Prediction': predictions})
# submission.to_csv("submission.csv", index=False)

