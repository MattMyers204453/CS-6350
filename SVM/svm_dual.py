import array
import os
import sys
import numpy as np
import pandas as pd
import read_data_svm as read
from scipy import optimize

	
attributes = ["variance", "skewness", "kurtosis", "entropy", "y"]
df = read.read_data_into_dataframe("bank-note/train.csv", attributes, 10000)
test_df = read.read_data_into_dataframe("bank-note/test.csv", attributes, 10000)
#test_df = df
df2 = df.copy()
sys.displayhook(df)
#del df["y"]
x_matrix = df.to_numpy()
# print(x_matrix)
y_matrix = df["y"].to_numpy()
# print(y_matrix)
#sys.displayhook(df2)
C = 100.0 / 872.0
alphas = [0] * (len(x_matrix))
# C_vector = [C] * (len(x_matrix))

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

def predict(w, x_vector):
    return sgn(np.dot(w, x_vector))

def dual_function(alphas, x_matrix, y_matrix):
    x_squared = np.matmul(x_matrix, x_matrix.T)
    y_squared = np.matmul(y_matrix, y_matrix.T)
    Hadamard_product = np.multiply(x_squared, y_squared)
    term1 = np.dot(alphas.T, Hadamard_product)
    term2 = np.dot(term1, alphas)
    result = 0.5 * term2 - np.sum(alphas)
    return result

# SVM Dual Optimization 
# num_of_features = len(attributes) - 1 # minus one to drop off the label 
# w_as_list = [0] * (num_of_features + 1) # plus one to account for constant in w-vector (notational sugar)
# w = np.array(w_as_list)
# N = len(df.index) ### THIS SHOULD BE 873, not 872!
# T = 100
# r = 0.005
# a = 2
# C = 100.0 / 872.0 ### THIS SHOULD BE 873, not 872!
#C = 500.0 / 872.0
#C = 700.0 / 872.0

con1 = {'type': 'eq', 'fun': lambda alphas: np.dot(alphas, y_matrix), 'jac': lambda alphas: y_matrix}
constraints = []
for i in range(len(alphas)):
    def con_func(i = i):
        return C - alphas[i]
    con = {'type': 'ineq', 'fun': con_func, 'jac': lambda alpha: -1}
    constraints.append(con)
# constraints = constraints+[con1]
constraints = constraints.append(con1)

optimal_params = optimize.minimize(fun=dual_function, args=(x_matrix, y_matrix), x0=alphas, method="SLSQP", constraints=constraints)
print(optimal_params.x)

# dat = np.array([[0, 3], [-1, 0], [1, 2], [2, 1], [3,3], [0, 0], [-1, -1], [-3, 1], [3, 1]])
# print(dat[2, :])

# l = [1, 2, 3]
# l = l+[4, 5]
# print(l)

# constraints = ({'type': 'eq', 'fun': lambda alphas: np.dot(alphas, y_matrix), 'jac': lambda a: y_matrix},{'type': 'ineq', 'fun': lambda alphas: b - np.dot(A, a), 'jac': lambda a: -A})

# print("Testing model on testing data...")
# errors = 0
# for i in range(len(test_df.index)):
#     row = test_df.iloc[i]
#     x_vector = get_x_vector_at(i, test_df)
#     actual = row.get("y")
#     guess = predict(w, x_vector) 
#     if guess != actual:
#         errors += 1

# print(f"TOTAL MISCLASSIFIED: {errors}")
# print(f"---TEST ACCURACY: {((float(len(test_df.index)) - errors) / float(len(test_df.index))) * 100}%")

# print("Testing model on training data...")
# errors = 0
# for i in range(len(df.index)):
#     row = df.iloc[i]
#     x_vector = get_x_vector_at(i, df)
#     actual = row.get("y")
#     guess = predict(w, x_vector) 
#     if guess != actual:
#         errors += 1

# print(f"TOTAL MISCLASSIFIED: {errors}")
# print(f"---TRAIN ACCURACY: {((float(len(df.index)) - errors) / float(len(df.index))) * 100}%")
# print(f"***MODEL PARAMETERS: {w}")