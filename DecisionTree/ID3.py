from sys import displayhook
import pandas as p
import math

def getp(df, label_value):
    total = len(df.index) #WRONG. SHOULD BETOTAL NUMBER OF POSSIBLE LABELS FOR THAT VALUE
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
        if prob_i != 0:
            sum += (prob_i * prob_i)
    return (1 - sum)

# feature refers to particular attribute
def get_entropy_of_feature_at_specific_value(df, feature, value, labels):
    total_num_features_at_that_value = sum(df[feature] == value)
    print(total_num_features_at_that_value)
    subset = df.loc[(df[feature] == value)] #& (df["label"] == "unacc")]
    displayhook(subset)
    print(len(subset))
    sigma = 0.0
    for label in labels:
        prob_i = getp(subset, label)
        if prob_i != 0:
            sigma += (prob_i * prob_i)
    print(1 - sum)



def get_IG(entropy_of_set, df, label):
    return 0


with open ('data-desc.txt') as file:
    lines = file.readlines()
file.close()
labels_as_csv_string = lines[2]
attributes_as_csv_string = lines[14]

labels = labels_as_csv_string.strip().split(", ")
attributes = attributes_as_csv_string.strip().split(",")

attribute_values = {}

for i in range(len(attributes)):
    uncut_line = lines[i + 6]
    substring = uncut_line[9:]
    attribute_values[attributes[i]] = substring.strip()[:-1].split(", ")

print(attribute_values)



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
#print(get_entropy_of_feature(df,))
get_entropy_of_feature_at_specific_value(df, "buying", "med", labels)
print(get_GI_of_dataset(df, labels))
