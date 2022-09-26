from sys import displayhook
import pandas as p
import math

def getp(df, label_value):
    total = len(df.index) 
    if (total == 0):
        return 0
    numerator = sum(df["label"] == label_value)
    return (numerator / float(total))



def get_entropy_of_dataset(df, labels):
    sum = 0.0
    for label in labels:
        prob_i = getp(df, label)
        if prob_i != 0:
            sum += (prob_i * math.log(prob_i, 2))
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
        #print("GI", str(GI))
        gini_for_each_value[value] = GI
    #max_value = max(gini_for_each_value, key=gini_for_each_value.get)
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
        print(attributes[i], " ",str(IG))
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

def ID3(df, attributes, attribute_values, labels, most_common_label):
    #gini_of_dataset = get_GI_of_dataset(df, labels)
    #if (gini_of_dataset == 0):
        #return 0
    if (len(attributes) == 0):
        return most_common_label
    for label in labels:
        if sum(df["label"] == label) == len(df.index):
            return Node(False, None, True, label)
    root = Node()
    feature_with_highest_IG = find_highest_IG(df, attributes, labels, attribute_values)
    i = 0
    for value in attribute_values[feature_with_highest_IG]:
        root.children[i] = Node()
        subset = df.loc[(df[feature_with_highest_IG] == value)]
        if (len(subset.index) == 0):
            most_common_label_in_df = df["label"].mode()
        i += 1

        

#with open ('data-desc copy.txt') as file:
with open ('data-desc.txt') as file:
    lines = file.readlines()
file.close()
labels_as_csv_string = lines[2]
attributes_as_csv_string = lines[14]

labels = labels_as_csv_string.strip().split(", ")
attributes = attributes_as_csv_string.strip().split(",")
#print(attributes)

attribute_values = {}

for i in range(len(attributes)):
    uncut_line = lines[i + 6]
    substring = uncut_line[9:]
    attribute_values[attributes[i]] = substring.strip()[:-1].split(", ")

##print(labels)
##print(attributes)
##print(attribute_values)




dict = {}
for i in range(len(attributes)):
    dict[attributes[i]] = []
#with open("test copy.csv") as file:
count = 0
with open("test.csv") as file:
    for line in file:
        if count > 5:
            break
        terms = line.strip().split(",")
        for i in range(len(attributes)):
            dict[attributes[i]].append(terms[i]) #= terms[i]
        count += 1


df = p.DataFrame(dict)
#print(dict)
displayhook(df)
#print(sum(df["label"] == "unacc"))
#print(len(df.index))
#print(getp(df, "unacc"))
#print(get_entropy_of_dataset(df, labels))
#print(get_entropy_of_feature(df,))
#get_GI_of_feature_at_specific_value(df, "outlook", "rainy", labels)
#IG_of_set = get_GI_of_dataset(df, labels)
#print(get_IG(IG_of_set, df, "outlook", attribute_values, labels))
#best_feature = find_highest_IG(df, attributes, labels, attribute_values)
#print(best_feature)
#print(ID3(df, attributes, attribute_values, labels, "unacc").label)
subset = df.loc[(df["label"] == "acc")]
if (len(subset.index) == 0):
    most_common_label_in_df = df["label"].mode()
    print(most_common_label_in_df)
