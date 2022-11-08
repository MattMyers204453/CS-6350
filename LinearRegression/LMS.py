
import sys
import pandas as p
import math
import read_data_LinearRegression as read
import os
import numpy as np

attributes = ["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "SLUMP"]

print("Reading data...")
df = read.read_data_into_dataframe("train.csv", attributes, 1000000)    
test_df = read.read_data_into_dataframe("test.csv", attributes, 100000)
#test_df = df
#####################################

def compute_gradient_of_cost(df, w, m):
    m = len(df.index)
    w_gradient = [1] * len(w)
    for j in range(len(w)):
        w_gradient[j] = compute_partial_gradient(df, w, j, m)
    return np.array(w_gradient)

def compute_partial_gradient(df, w, j, m):
    sum = 0.0
    for i in range(m):
        y_i = get_label_at(i, df)
        x_i = get_x_vector_at(i, df)
        term = (y_i - (np.dot(w.T, x_i))) * x_i[j]
        sum += term
    return sum * -1


def sum_squares(df, w):
    return 0

def get_x_vector_at(i, df):
    row = df.iloc[i]
    row_as_list = row.to_list()[:-1]
    row_as_list.insert(0, 1)
    row_as_matrix = np.array(row_as_list)
    return row_as_matrix

def get_label_at(i, df):
    row = df.iloc[i]
    return row.get("SLUMP")



#####################################
num_of_features = len(attributes) - 1 # minus one to drop off the label (SLUMP)
w_as_list = [0] * (num_of_features + 1) # plus one to account for constant in w-vector (notational sugar)
w = np.array(w_as_list)
m = len(df.index)


T = int(sys.argv[1])
r = float(sys.argv[2])

# T = 2
# r = 0.002
print(f"T = {T}")
print(f"r = {r}")
print("Training model...")
for t in range(T):
    gradient = compute_gradient_of_cost(df, w, m)
    w = np.subtract(w, r * gradient)

print("Testing model...")
squared_error_sum = 0
size = len(test_df.index)
for i in range(size):
    row = test_df.iloc[i]
    actual_value = row.get("SLUMP")
    x_vector = get_x_vector_at(i, test_df)
    guess = np.dot(w.T, x_vector)
    diff = abs(actual_value - guess)
    squared_error = diff * diff
    # print(f"ACTUAL: {actual_value} GUESS: {guess}")
    squared_error_sum += squared_error
MSE = squared_error_sum / float(size)
print(f"MEAN SQUARED ERROR: {MSE}")

