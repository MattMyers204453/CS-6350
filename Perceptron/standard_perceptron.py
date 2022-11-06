# import pandas module
import os
import sys
import numpy as np
import pandas as pd
import read_data_perceptron as read

	
attributes = ["variance", "skewness", "kurtosis", "entropy", "y"]
df = read.read_data_into_dataframe("bank-note/train.csv", attributes, 1000)
test_df = read.read_data_into_dataframe("bank-note/test.csv", attributes, 1000)

# sys.displayhook(df)

def get_x_vector_at(i, df):
    row = df.iloc[i]
    row_as_list = row.to_list()[:-1]
    # row_as_list.insert(0, 1)
    row_as_list.append(1)
    row_as_matrix = np.array(row_as_list)
    # print(row_as_matrix)
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

#Perceptron
num_of_features = len(attributes) - 1 # minus one to drop off the label 
w_as_list = [0] * (num_of_features + 1) # plus one to account for constant in w-vector (notational sugar)
w = np.array(w_as_list)
m = len(df.index)

r = 1
for j in range(10):
    df = df.sample(frac=1)
    for i in range(len(df.index)):
        row = df.iloc[i]
        x_vector = get_x_vector_at(i, df)
        actual_value = row.get("y")
        prediction = predict(w, x_vector)
        if prediction != actual_value:
            w = update(w, actual_value, x_vector, r)
        # print(w)

errors = 0
for i in range(len(test_df.index)):
    row = test_df.iloc[i]
    x_vector = get_x_vector_at(i, test_df)
    actual = row.get("y")
    guess = predict(w, x_vector) 
    # print(actual, " ", guess)   
    if guess != actual:
        errors += 1

print(errors)
print(f"ACCURACY: {(float(len(test_df.index)) - errors) / float(len(test_df.index))}")
