#from sys import displayhook
import sys
import pandas as p
import math
import read_data_ensemble as read

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
df = read.read_data_into_dataframe("train.csv", attributes, 1000000)
#sys.displayhook(df)

df = read.convert_dataframe(df)
sys.displayhook(df)

##################################################################



### DECISION TREE FUNCTIONS ###
def getp(df, label_value):
    total = len(df.index) 
    if (total == 0):
        return 0
    numerator = sum(df["y"] == label_value)
    return (numerator / float(total))

def get_entropy_of_dataset(df, labels):
    sum = 0.0
    for label in labels:
        prob_i = getp(df, label)
        if prob_i != 0:
            sum += (prob_i * math.log(prob_i, len(labels)))
    return sum * -1

def get_GI_of_dataset(df, labels):
    sum = 0.0
    for label in labels:
        prob_i = getp(df, label)
        sum += (prob_i * prob_i)
    return (1 - sum)

# feature refers to particular attribute
def get_GI_of_feature_at_specific_value(df, feature, value, labels):
    #total_num_features_at_that_value = sum(df[feature] == value)
    subset = df.loc[(df[feature] == value)] #& (df["label"] == "unacc")]
    if len(subset.index) == 0:
        return 0
    #displayhook(subset)
    sigma = 0.0
    for label in labels:
        prob_i = getp(subset, label)
        sigma += (prob_i * prob_i)
    return (1 - sigma)


# feature refers to particular attribute
def get_IG(GI_of_set, df, feature, attribute_values, labels):
    gini_for_each_value = {}
    for value in attribute_values[feature]:
        GI = get_GI_of_feature_at_specific_value(df, feature, value, labels)
        gini_for_each_value[value] = GI
    length_of_whole_set = len(df.index)
    sigma = 0.0
    for value, GI in gini_for_each_value.items():
        num_features_at_this_value = sum(df[feature] == value)
        term = (num_features_at_this_value / float(length_of_whole_set)) * GI
        sigma += term
    return GI_of_set - sigma

def find_highest_IG(df, attributes, labels, attribute_values, weights):
    gini_of_set = get_GI_of_dataset(df, labels)
    IG_for_each_value = {}
    for i in range(len(attributes) - 1):
        IG = get_IG(gini_of_set, df, attributes[i], attribute_values, labels)
        IG_for_each_value[attributes[i]] = IG
    best_feature = max(IG_for_each_value, key=IG_for_each_value.get)
    return best_feature

class Node:
    def __init__(self):
        pass
    def __init__(self, isFeatureNode, feature, isLeafNode, label):
        self.isFeatureNode = isFeatureNode
        self.isLeafNode = isLeafNode
        self.children = {}
        self.feature = feature
        self.label = label

def traverse_tree(i, df, root, attribute_values):
    while root.isLeafNode == False:
        feature = root.feature
        item_value = df.at[i, feature]
        index = attribute_values[feature].index(item_value)
        root = root.children[index]
    return root.label
#########################################################################

### BOOSTING FUNCTIONS ###
def get_stump(df, attributes, attribute_values, labels):
    feature_with_highest_IG = find_highest_IG(df, attributes, labels, attribute_values)
    root = Node(True, feature_with_highest_IG, False, None)
    i = 0
    for value in attribute_values[feature_with_highest_IG]:
        subset = df.loc[(df[feature_with_highest_IG] == value)]
        weak_stump_guess = None
        if (len(subset.index) == 0):
            weak_stump_guess = df["y"].mode()[0]
        else:
            weak_stump_guess = subset["y"].mode()[0]
        leaf = Node(False, None, True, weak_stump_guess)
        root.children[i] = leaf
        i += 1
    return root
##########################################################################

def test_then_get_alpha_and_agreement_vector(stump, test_df, weights, num_test_examples):
    agreement_vector = [1] * num_test_examples
    error = 0.0
    for i in range(len(test_df.index)):
        row = test_df.iloc[[i]]
        actual_label = row.at[i, "y"]
        result_label = traverse_tree(i, row, stump, attribute_values)
        if (actual_label != result_label):
            error += weights[i]
            agreement_vector[i] = -1
    alpha = (1 / 2) * math.log((1 - error) / error)
    return (alpha, agreement_vector)

def reweight(old_weights, alpha, agreement_vector):
    new_weights_unormalized = [0] * len(old_weights)
    for i in range(len(new_weights_unormalized)):
        new_weights_unormalized[i] = old_weights[i] * math.exp(-1 * alpha * agreement_vector[i])
    normalization_constant = sum(new_weights_unormalized)
    new_weights = [0] * len(old_weights)
    for i in range(len(new_weights)):
        new_weights[i] = new_weights_unormalized[i] / normalization_constant
    return new_weights


#--------MAIN ---------------------------------------------------------------------------------------------------------------#
### ADABOOST ###
t = 10
alpha_values = [0] * t
weak_classifier = [None] * 10
test_df = read.read_data_into_dataframe("test.csv", attributes, 1000000)
test_df = read.convert_dataframe(test_df)
num_test_examples = len(test_df.index)
weights = [1 / num_test_examples] * num_test_examples
for i in range(t):
    ### Obtain weak classifier ###
    stump = get_stump(df, attributes, attribute_values, labels)

    ### Get vote (and "agreement vector") ###
    alpha_and_agreement_vector = test_then_get_alpha_and_agreement_vector(stump, test_df, weights)
    alpha = alpha_and_agreement_vector[0]
    agreement_vector = alpha_and_agreement_vector[1]

    ### Record alpha value or "vote" for this round ###
    alpha_values[i] = alpha



# error_count = 0.0
# for i in range(len(test_df.index)):
#     row = test_df.iloc[[i]]
#     actual_label = item_value = row.at[i, "y"]
#     result_label = traverse_tree(i, row, stump, attribute_values)
#     if (actual_label != result_label):
#         error_count += 1
# print("Total Errors: ", str(error_count))
# print("Accuracy: ", (float(len(test_df.index)) - float(error_count)) / float(len(test_df.index)))
# print(stump.feature)

#---------------------------------------------------------------------------------------------------------------------------#