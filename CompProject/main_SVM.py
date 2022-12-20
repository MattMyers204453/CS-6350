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

def read_data_into_dataframe_SVM(file_name, attributes, test, max):
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

def preprocess_dataframe_SVM(df, attributes, attribute_values, test):
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
    if not test:
        map = {1: 1, 0: -1}
        df["income>50K"] = df["income>50K"].astype("int").map(map)
        label_column_to_move = df.pop("income>50K")
        df.insert(len(df.columns), "income>50K", label_column_to_move)



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
            df.at[i, new_category] = 1 if value_with_1 == new_category else -1
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

labels = [0, 1]

attributes = ["age","workclass","fnlwgt","education","education.num","marital.status","occupation", "relationship","race", "sex","capital.gain","capital.loss","hours.per.week","native.country","income>50K"]
df = read_data_into_dataframe_SVM(TRAIN_PATH, attributes, False, 150)
preprocess_dataframe_SVM(df, attributes, attribute_values, False)
#sys.displayhook(df)

# SVM
##################
def get_x_vector_at_DATA_HAS_NO_LABEL(i, df):
    row = df.iloc[i]
    row_as_list = row.to_list()
    row_as_list.append(1)
    row_as_matrix = np.array(row_as_list)
    return row_as_matrix

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
    
new_attribute_names = list(df.columns.values)
new_attribute_names.remove("income>50K")
x_matrix = df[new_attribute_names].to_numpy()
y_matrix = df["income>50K"].to_numpy()
M, N = x_matrix.shape
alphas = [0] * (len(x_matrix))
C = 100.0 / 872.0
#C = 500.0 / 872.0
#C = 700.0 / 872.0

XXYY = compute_element_wise_product()
# XXYY = compute_element_wise_product_kernel()
constraints = ({'type': 'eq', 'fun': lambda alphas: np.dot(alphas, y_matrix), 'jac': lambda alphas: y_matrix})
bounds= Bounds(np.zeros(M), np.full(M, C))

print("Training model...")
optimal_alphas = optimize.minimize(fun=dual_function, args=(x_matrix, y_matrix, XXYY), x0=alphas, method="SLSQP", constraints=constraints, bounds=bounds)
w_0 = recover_weight_vector(optimal_alphas.x, x_matrix, y_matrix, M, N)
b = recover_b(optimal_alphas.x, x_matrix, y_matrix, w_0, C)
w = np.append(w_0, b)

print("Testing model on training data...")
errors = 0
for i in range(len(df.index)):
    row = df.iloc[i]
    x_vector = get_x_vector_at(i, df)
    actual = row.get("income>50K")
    guess = predict(w, x_vector) 
    if guess != actual:
        errors += 1

print(f"TOTAL MISCLASSIFIED: {errors}")
print(f"---TRAIN ACCURACY: {((float(len(df.index)) - errors) / float(len(df.index))) * 100}%")
#print(f"***MODEL PARAMETERS: {w}")
##################


print("Reading test data...")
test_df = read_data_into_dataframe_SVM(TEST_PATH, attributes[:-1], True, 10)
print("Preprocessing test data...")
preprocess_dataframe_SVM(test_df, attributes[:-1], attribute_values, True)
predictions = [0] * len(test_df)

print("Testing model on test data...")
error_count = 0
for i in range(len(test_df.index)):
    row = test_df.iloc[i]
    x_vector = get_x_vector_at_DATA_HAS_NO_LABEL(i, test_df)
    guess = predict(w, x_vector)
    predictions[i] = guess

print("Writing predictions to file...")
#test = p.read_csv(os.path.join(sys.path[0], 'test_final.csv'))
ID_s = [1] * len(test_df)
for i in range(len(test_df)):
    ID_s[i] = i + 1
submission = p.DataFrame({'Id': ID_s, 'Prediction': predictions})
submission.to_csv("submission.csv", index=False)

