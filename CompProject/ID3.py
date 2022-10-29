#from sys import displayhook
import sys
import pandas as p
import math
import os


def getp(df, label_value):
    total = len(df.index) 
    if (total == 0):
        return 0
    numerator = sum(df["income>50K"] == label_value)
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
    return get_GI_of_dataset(subset, labels)


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

def find_highest_IG(df, attributes, labels, attribute_values):
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

def ID3(df, attributes, attribute_values, labels):
    most_common_label = df["income>50K"].mode()[0]
    if (len(attributes) == 1):
        return Node(False, None, True, most_common_label)
    for label in labels:
        if sum(df["income>50K"] == label) == len(df.index):
            return Node(False, None, True, label)
    feature_with_highest_IG = find_highest_IG(df, attributes, labels, attribute_values)
    root = Node(True, feature_with_highest_IG, False, None)
    i = 0
    for value in attribute_values[feature_with_highest_IG]:
        #root.children[i] = Node()
        subset = df.loc[(df[feature_with_highest_IG] == value)]
        if (len(subset.index) == 0):
            most_common_label_in_df = df["income>50K"].mode()[0]
            leaf = Node(False, None, True, most_common_label_in_df)
            root.children[i] = leaf
        else:
            subAttributes = attributes[:]
            subAttributes.remove(feature_with_highest_IG)
            subAttribute_values = attribute_values.copy()
            del subAttribute_values[feature_with_highest_IG]
            subtree = ID3(subset, subAttributes, subAttribute_values, labels)
            root.children[i] = subtree
        i += 1
    return root

def traverse_tree(row, root, attribute_values):
    while root.isLeafNode == False:
        feature = root.feature
        item_value = row.get(feature)
        index = attribute_values[feature].index(item_value)
        root = root.children[index]
    return root.label


