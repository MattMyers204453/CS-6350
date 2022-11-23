import os
import sys
import numpy as np
import pandas as pd
import read_data_svm as read

	
attributes = ["variance", "skewness", "kurtosis", "entropy", "y"]
df = read.read_data_into_dataframe("bank-note/train.csv", attributes, 10000)
#test_df = read.read_data_into_dataframe("bank-note/test.csv", attributes, 10000)
test_df = df

#sys.displayhook(df)

def get_x_vector_at(i, df):
    row = df.iloc[i]
    row_as_list = row.to_list()[:-1]
    row_as_list.append(1)
    row_as_matrix = np.array(row_as_list)
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

# SVM Sub-gradient descent 
num_of_features = len(attributes) - 1 # minus one to drop off the label 
w_as_list = [0] * (num_of_features + 1) # plus one to account for constant in w-vector (notational sugar)
w = np.array(w_as_list)
N = len(df.index) ### THIS SHOULD BE 873, not 872!
T = 100
r = 0.005
a = 2
#C = 100.0 / 872.0 ### THIS SHOULD BE 873, not 872!
C = 500.0 / 872.0
#C = 100.0 / 872.0
##################w[len(w) - 1] = 0

# print(sys.argv)
# a = [0] * (len(w_as_list))
# T = int(sys.argv[1]) if len(sys.argv) == 3 else 10
# r = float(sys.argv[2]) if len(sys.argv) == 3 else 1.0
print(f"Epochs: {T}")
print(f"Learning rate = {r}")
print("Training model...")
for t in range(T):
    r = r / float(1 + (r*t) / float(a))
    df = df.sample(frac=1)
    for i in range(len(df.index)):
        row = df.iloc[i]
        x_i = get_x_vector_at(i, df)
        y_i = row.get("y")
        if (y_i * np.dot(w, x_i)) <= 1:
            term1 = r * w
            term1[len(term1) - 1] = 0
            w = w - term1 + (r * C * N * y_i * x_i)
        else:
            saved_b = w[len(w) - 1]
            w = (1 - r) * w
            w[len(w) - 1] = saved_b
print(w)

print("Testing model...")
errors = 0
for i in range(len(test_df.index)):
    row = test_df.iloc[i]
    x_vector = get_x_vector_at(i, test_df)
    actual = row.get("y")
    guess = predict(w, x_vector) 
    if guess != actual:
        errors += 1

print(f"TOTAL MISCLASSIFIED: {errors}")
print(f"ACCURACY: {((float(len(test_df.index)) - errors) / float(len(test_df.index))) * 100}%")