import sys
from turtle import pos
import pandas as p
import math
from ID3 import ID3, traverse_tree
import read_data_ensemble as read
import numpy as np
import os

### Attributes, categorical attributes, and labels ###
attributes = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
              "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]
cat_attributes = ["job", "marital", "education", "contact", "month", "poutcome"]
labels = [-1, 1]
##########################################################

### Dictionary of possible values for each attribute ###
attribute_values = {}
attribute_values["age"] = [0, 1]
attribute_values["default"] = [0, 1]
attribute_values["balance"] = [0, 1]
attribute_values["housing"] = [0, 1]
attribute_values["loan"] = [0, 1]
attribute_values["day"] = [0, 1]
attribute_values["duration"] = [0, 1]
attribute_values["campaign"] = [0, 1]
attribute_values["pdays"] = [0, 1]
attribute_values["previous"] = [0, 1]
attribute_values["job"] = ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services"]
attribute_values["marital"] = ["married","divorced","single"]
attribute_values["education"] = ["unknown","secondary","primary","tertiary"]
attribute_values["contact"] = ["unknown","telephone","cellular"]
attribute_values["month"] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
attribute_values["poutcome"] = ["unknown","other","failure","success"]
attribute_values["y"] = [-1, 1]
###########################################################

### Read train.csv and convert into dataframe. Convert numerical column values from strings to ints. Convert numerical data to boolean data ###
df = read.read_data_into_dataframe("train.csv", attributes, 100000)
df = read.convert_dataframe(df)
sys.displayhook(df)

test_df = read.read_data_into_dataframe("test.csv", attributes, 100000)
test_df = read.convert_dataframe(test_df)
##################################################################

def get_sample(df, size):
    df_random = df[0:0]
    for i in range(size):
        random_row = df.sample()
        df_random = p.concat([df_random, random_row], ignore_index=False)
    return df_random

def test_then_get_alpha(tree, df):
    num_test_examples = len(df.index)
    error_count = 0
    for i in range(num_test_examples):
        row = df.iloc[i]
        actual_label = row.get("y")
        result_label = traverse_tree(row, tree, attribute_values)
        if (actual_label != result_label):
            error_count += 1
    error = error_count / num_test_examples
    if error == 0:
        error = 0.00000001
    alpha = (1 / 2) * math.log((1 - error) / error)
    return alpha

def bagging_train(df, t, sample_length):
    alphas = [1] * t
    trees = [None] * t
    #m = len(df.index)
    for i in range(t):
        os.system('cls')
        print(f"ROUND: {i}")
        df_sample = get_sample(df, sample_length)
        root = ID3(df_sample, attributes, attribute_values, labels)
        trees[i] = root
        # alpha = test_then_get_alpha(root, df_sample)
        # alphas[i] = alpha
    return (trees, alphas)

def bagging_train(df_sample, trees, i):
    trees[i] = i * 10
    #alphas = [1] * t
    m = len(df.index)
    # df_sample = get_sample(df, m)
    root = ID3(df_sample, attributes, attribute_values, labels)
    trees[i] = root

def get_average_prediction(row, trained_model, attribute_values):
    trees = trained_model[0]
    alphas = trained_model[1]
    sum = 0.0
    for i in range(len(trees)):
        prediction = traverse_tree(row, trees[i], attribute_values)
        #sum += (prediction * alphas[i])
        sum += prediction
    if sum > 0:
        return 1
    return -1

def test_bagging(test_df, trained_model):
    error_count = 0
    for i in range(len(test_df.index)):
        row = test_df.iloc[i]
        actual_label = row.get("y")
        result_label = get_average_prediction(row, trained_model, attribute_values)
        if (actual_label != result_label):
            error_count += 1
    print("Total Errors: ", str(error_count))
    print("Accuracy: ", (float(len(test_df.index)) - float(error_count)) / float(len(test_df.index)))

### TEST ###
from threading import Thread

def get_trained_model_threading(df, T, sample_length):
    trees = [1] * T
    threads = []
    for i in range(T):
        df_sample = get_sample(df, sample_length)
        th = Thread(target=bagging_train, args=(df_sample, trees, i))
        threads.append(th)
    for th in threads:
        print("started")
        th.start()
    for th in threads:
        th.join()
        print("joined")

    print(len(trees))
    return (trees, [1] * T)

model_bagging = get_trained_model_threading(df, 100, 500)
#model_bagging = bagging_train(df, 20, 50)
test_bagging(test_df, model_bagging)



# import matplotlib.pyplot as plt
   
# T = [2, 5, 10, 50, 200]
# Accuracy = [0.8832, 0.8832, 0.8886, 0.8916, 0.9001]
  
# plt.plot(T, Accuracy, color='red', marker='o')
# plt.title('Accuracy for each T', fontsize=14)
# plt.xlabel('T', fontsize=14)
# plt.ylabel('Accuracy', fontsize=14)
# plt.grid(True)
# plt.show()
