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

def final_predict(w_c, x_vector):
    sum = 0
    for i in range(len(w_c)):
        w = w_c[i][0]
        c = w_c[i][1]
        sum += (c * sgn(np.dot(w, x_vector)))
    return sgn(sum)

#Perceptron
num_of_features = len(attributes) - 1 # minus one to drop off the label 
w_as_list = [0] * (num_of_features + 1) # plus one to account for constant in w-vector (notational sugar)
w = np.array(w_as_list)
m = len(df.index)

w_c = []
c = 1
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
        actual_value = row.get("y")
        prediction = predict(w, x_vector)
        if prediction != actual_value:
            w_c.append((w, c))
            c = 1
            w = update(w, actual_value, x_vector, r)
        else:
            c += 1

print("Testing model...")
errors = 0
for i in range(len(test_df.index)):
    row = test_df.iloc[i]
    x_vector = get_x_vector_at(i, test_df)
    actual = row.get("y")
    guess = final_predict(w_c, x_vector) 
    if guess != actual:
        errors += 1

print(f"TOTAL MISCLASSIFIED: {errors}")
print(f"ACCURACY: {(float(len(test_df.index)) - errors) / float(len(test_df.index))}")

# for i in range(len(w_c)):
#     print(w_c[i])
