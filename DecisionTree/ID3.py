from sys import displayhook
import pandas as p
import math

def getp(df, label_value):
    total = len(df.index)
    numerator = sum(df["label"] == label_value)
    return (numerator / float(total))


def get_entropy_of_dataset(df, labels):
    sum = 0.0
    for label in labels:
        prob_i = getp(df, label)
        if prob_i != 0:
            sum += (prob_i * math.log(prob_i, 2))
    return sum * -1

# feature refers to particular attribute
def get_entropy_of_feature_at_specific_value(df, value, feature, labels, attributes):
    total_num_features_at_that_value = 


def get_IG(entropy_of_set, df, label):
    return 0


with open ('data-desc.txt') as file:
    lines = file.readlines()
file.close()
labels_as_csv_string = lines[2]
attributes_as_csv_string = lines[14]

labels = labels_as_csv_string.strip().split(", ")
attributes = attributes_as_csv_string.strip().split(",")

dict = {}
for i in range(len(attributes)):
    dict[attributes[i]] = []
count = 0;
with open("test.csv") as file:
    for line in file:
        if count > 9:
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
print(get_entropy_of_feature(df,))
