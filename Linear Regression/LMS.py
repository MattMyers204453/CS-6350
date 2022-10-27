import sys
import pandas as p
import math
import read_data_LinearRegression as read
import os

attributes = ["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "SLUMP"]

df = read.read_data_into_dataframe("test.csv", attributes, 1000000)
sys.displayhook(df)

sys.displayhook(df)
#####################################

def compute_gradient_of_cost(df, w, m):
    for i in range(len(w)):
        break
    return -1 
    

def sum_squares(df, w):
    return 0

def get_x_vector_at(i, df):
    row = df.iloc[i]
    row_as_list = row.to_list()[:-1]
    row_as_list.insert(0, 1)
    return row_as_list

def get_label_at(i, df):
    row = df.iloc[i]
    return row.get("SLUMP")



#####################################
num_of_features = len(attributes) - 1 # minus one to drop off the label (SLUMP)
w = [0] * (num_of_features + 1) # plus one to account for constant in w-vector (notational sugar)
m = len(df.index)

print(get_x_vector_at(0, df))
print(get_label_at(5, df))
### GRADIENT DESCENT ###
# T = 10
# for t in range(T):

