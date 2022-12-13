import sys
import numpy as np
import pandas as pd
import read_data_neural as read
import math

attributes = ["variance", "skewness", "kurtosis", "entropy", "y"]
df = read.read_data_into_dataframe("bank-note/train.csv", attributes, 10000)
test_df = read.read_data_into_dataframe("bank-note/test.csv", attributes, 10000)
sys.displayhook(df)

##############################################
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
##############################################

ONE = 1
TWO = 2
THREE = 3
numFeatures = len(attributes) - 1
num_of_zs_at_layer_2 = 3#5
num_of_zs_at_layer_1 = 3#5
num_of_xs_at_layer_0 = 3#numFeatures + 1
weights_dict = {}
z_dict = {}

def initialize_weights_paper_problem():
    weights_dict[w(0, 1, 3)] = -1.0
    weights_dict[w(1, 1, 3)] =  2.0
    weights_dict[w(2, 1, 3)] = -1.5

    weights_dict[w(0, 1, 2)] = -1.0
    weights_dict[w(0, 2, 2)] = 1.0
    weights_dict[w(1, 1, 2)] = -2.0
    weights_dict[w(1, 2, 2)] = 2.0
    weights_dict[w(2, 1, 2)] = -3.0
    weights_dict[w(2, 2, 2)] = 3.0

    weights_dict[w(0, 1, 1)] = -1.0
    weights_dict[w(0, 2, 1)] = 1.0
    weights_dict[w(1, 1, 1)] = -2.0
    weights_dict[w(1, 2, 1)] = 2.0
    weights_dict[w(2, 1, 1)] = -3.0
    weights_dict[w(2, 2, 1)] = 3.0


def initialize_weights():
    # layer 3
    for i in range(num_of_zs_at_layer_2):
        weights_dict[w(i, ONE, THREE)] = np.random.normal(0, 1)
    # layer 2
    for n in range(num_of_zs_at_layer_2):
        for m in range(1, num_of_zs_at_layer_1):
            weights_dict[w(m, n, TWO)] = np.random.normal(0, 1)
    # layer 1
    for n in range(num_of_zs_at_layer_1):
        for m in range(1, num_of_xs_at_layer_0):
            weights_dict[w(m, n, ONE)] = np.random.normal(0, 1)

def w(start, to, layer):
    return f"w_{start}{to}^{layer}"

def z(num, layer):
    return f"z_{num}^{layer}"

def loss_partial(y, y_star):
    return y - y_star

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-1 * x))

def sigmoid_partial(s):
    return sigmoid(s) * (1 - sigmoid(s))

def get_s_for_given_layer2_z(z_num):
    s = 0
    for i in range(num_of_zs_at_layer_1):
        s += (weights_dict[w(i, z_num, TWO)] * z_dict[z(i, ONE)])
    return s

def get_s_for_given_layer1_z(z_num, x_input):
    s = 0
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
            grad_weights_layer_2[w(m, ONE, TWO)] = L_partial * weights_dict[w(n, ONE, THREE)] * sig_partial * z_dict[z(m, ONE)]
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
        outside_s = get_s_for_given_layer1_z(n)
        term_outside_bracket = sig_partial(outside_s)
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

def back_propagation(y, y_star):
    w_layer_3_partials = get_layer_3_partials(y, y_star)
    w_layer_2_partials = get_layer_2_partials(y, y_star)
    w_layer_1_partials = get_layer_1_partials(y, y_star)
    return (w_layer_1_partials, w_layer_2_partials, w_layer_3_partials)

def update_weights(r, w_partials_tuple):
    w_layer_3_partials = w_partials_tuple[2]
    w_layer_2_partials = w_partials_tuple[1]
    w_layer_1_partials = w_partials_tuple[0]
    # layer 3
    for i in range(num_of_zs_at_layer_2):
        weights_dict[w(i, ONE, THREE)] -= r * w_layer_3_partials[w(i, ONE, THREE)]
    # layer 2
    for n in range(num_of_zs_at_layer_2):
        for m in range(1, num_of_zs_at_layer_1):
            weights_dict[w(m, n, TWO)] -= r * w_layer_2_partials[w(m, n, TWO)]
    # layer 1
    for n in range(num_of_zs_at_layer_1):
        for m in range(1, num_of_xs_at_layer_0):
            weights_dict[w(m, n, ONE)] -= r * w_layer_1_partials[w(m, n, ONE)]


##############################################################################################################################
#x_input_paper_problem = [1, 1, 1, 1]

# row = df.iloc[0]
# x_vector = get_x_vector_at(0, df)
# x_vector = np.insert(x_vector, 0, 1)[:-1]
# actual = row.get("y")
# print(x_vector)
# print(actual)

initialize_weights()

# print(f"Epochs: {T}")
# print(f"Learning rate = {r}")
# print("Training model...")

T = 3
r = 0.005
d= 0.5
for t in range(T):
    df = df.sample(frac=1)
    r = r / float(1 + (r * t) / float(d))
    for i in range(len(df.index)):
        row = df.iloc[i]
        x_i = get_x_vector_at(i, df)
        x_input = np.insert(x_i, 0, 1)[:-1]
        y_star = row.get("y")
        prediction = forward_pass(x_input)
        if y_star != sgn(prediction):
            gradient_of_loss = back_propagation(y=prediction, y_star=y_star)
            update_weights(r, gradient_of_loss)


