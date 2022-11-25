import array
import os
import sys
import numpy as np
import pandas as pd
import read_data_svm as read
from scipy import optimize
from scipy.optimize import Bounds
import math
	
attributes = ["variance", "skewness", "kurtosis", "entropy", "y"]
df = read.read_data_into_dataframe("bank-note/train.csv", attributes, 10000)
test_df = read.read_data_into_dataframe("bank-note/test.csv", attributes, 10000)
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

def predict(w, x_vector):
    return sgn(np.dot(w, x_vector))

def kernel(x_i, x_j, gamma):
    numerator = -1 * (np.linalg.norm((x_i - x_j), ord=1))**2
    return math.exp(numerator / gamma)

def compute_element_wise_product():
    XXYY = np.zeros((M, M))
    for r in range(M):
        for c in range(M):
            XXYY[r,c] = np.dot(x_matrix[r,:], x_matrix[c,:]) * y_matrix[r] * y_matrix[c]
    return XXYY

def compute_element_wise_product_kernel():
    XXYY = np.zeros((M, M))
    for r in range(M):
        for c in range(M):
            XXYY[r,c] = kernel(x_matrix[r,:], x_matrix[c,:]) * y_matrix[r] * y_matrix[c]
    return XXYY

def dual_function(alphas, x_matrix, y_matrix, XXYY):
    term1 = np.dot(alphas.T, XXYY)
    term2 = np.dot(term1, alphas)
    result = 0.5 * term2 - np.sum(alphas)
    return result

def recover_weight_vector(optimal_alphas, x_matrix, y_matrix, M, N):
    w = np.zeros(N)
    for i in range(M):
        w = w + (optimal_alphas[i] * y_matrix[i] * x_matrix[i, :])        
    return w

def recover_b(optimal_alphas, x_matrix, y_matrix, w, C):
    indices = np.where((optimal_alphas > 0.0000001)&(optimal_alphas < C))[0]
    sum = 0.0
    for j in indices:
        sum += y_matrix[j] - np.dot(w, x_matrix[j])
    b = sum / float(len(indices))
    return b
    # Uncomment below to use first on-the-margin example to recover b, rather than the average
    # j = indices[0] 
    # return y_matrix[j] - np.dot(w, x_matrix[j])

def recover_weight_vector_kernel(optimal_alphas, x_matrix, y_matrix, M, N):
    w = np.zeros(N)
    for i in range(M):
        w = w + (optimal_alphas[i] * y_matrix[i] * x_matrix[i, :])        
    return w

def recover_b_kernel(optimal_alphas, x_matrix, y_matrix, w, C):
    indices = np.where((optimal_alphas > 0.0000001)&(optimal_alphas < C))[0]
    sum = 0.0
    for j in indices:
        sum += y_matrix[j] - np.dot(w, x_matrix[j])
    b = sum / float(len(indices))
    return b

# SVM Dual Optimization 
x_matrix = df[["variance", "skewness", "kurtosis", "entropy"]].to_numpy()
y_matrix = df["y"].to_numpy()
M, N = x_matrix.shape
alphas = [0] * (len(x_matrix))
#C = 100.0 / 872.0
#C = 500.0 / 872.0
#C = 700.0 / 872.0
C = 1.1

# XXYY = compute_element_wise_product()
XXYY = compute_element_wise_product_kernel()
constraints = ({'type': 'eq', 'fun': lambda alphas: np.dot(alphas, y_matrix), 'jac': lambda alphas: y_matrix})
bounds= Bounds(np.zeros(M), np.full(M, C))

print("Training model...")
optimal_alphas = optimize.minimize(fun=dual_function, args=(x_matrix, y_matrix, XXYY), x0=alphas, method="SLSQP", constraints=constraints, bounds=bounds)
w_0 = recover_weight_vector(optimal_alphas.x, x_matrix, y_matrix, M, N)
b = recover_b(optimal_alphas.x, x_matrix, y_matrix, w_0, C)
w = np.append(w_0, b)

print("Testing model on testing data...")
errors = 0
for i in range(len(test_df.index)):
    row = test_df.iloc[i]
    x_vector = get_x_vector_at(i, test_df)
    actual = row.get("y")
    guess = predict(w, x_vector) 
    if guess != actual:
        errors += 1

print(f"TOTAL MISCLASSIFIED: {errors}")
print(f"---TEST ACCURACY: {((float(len(test_df.index)) - errors) / float(len(test_df.index))) * 100}%")

print("Testing model on training data...")
errors = 0
for i in range(len(df.index)):
    row = df.iloc[i]
    x_vector = get_x_vector_at(i, df)
    actual = row.get("y")
    guess = predict(w, x_vector) 
    if guess != actual:
        errors += 1

print(f"TOTAL MISCLASSIFIED: {errors}")
print(f"---TRAIN ACCURACY: {((float(len(df.index)) - errors) / float(len(df.index))) * 100}%")
print(f"***MODEL PARAMETERS: {w}")


