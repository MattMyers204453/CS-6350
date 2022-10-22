#from sys import displayhook
import sys
import pandas as p
import math
import read_data_ensemble as read

### Attributes and categorical attributes ###
attributes = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
              "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]
cat_attributes = ["job", "marital", "education", "contact", "month", "poutcome"]
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
attribute_values["job"] = ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services"]
attribute_values["marital"] = ["married","divorced","single"]
attribute_values["education"] = ["unknown","secondary","primary","tertiary"]
attribute_values["contact"] = ["unknown","telephone","cellular"]
attribute_values["month"] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
attribute_values["poutcome"] = ["unknown","other","failure","success"]
###########################################################

### Read train.csv and convert into dataframe. Convert numerical column values from strings to ints. Convert numerical data to booleans ###
df = read.read_data_into_dataframe("train.csv", attributes)
sys.displayhook(df)

df['age'] = df['age'].astype('int')
df['balance'] = df['balance'].astype('int')
df['day'] = df['day'].astype('int')
df['duration'] = df['duration'].astype('int')
df['campaign'] = df['campaign'].astype('int')
df['pdays'] = df['pdays'].astype('int')
df['previous'] = df['previous'].astype('int')

age_median = df['age'].median()
print(age_median)
balance_median = df['balance'].median()
print(balance_median)
day_median = df['day'].median()
print(day_median)
duration_median = df['duration'].median()
print(duration_median)
campaign_median = df['campaign'].median()
print(campaign_median)
pdays_median = df['pdays'].median()
print(pdays_median)
previous_median = df['previous'].median()
print(previous_median)


df['age'] = df['age'].gt(age_median).astype(int)
df['balance'] = df['balance'].gt(balance_median).astype(int)
df['day'] = df['day'].gt(day_median).astype(int)
df['duration'] = df['duration'].gt(duration_median).astype(int)
df['campaign'] = df['campaign'].gt(campaign_median).astype(int)
df['pdays'] = df['pdays'].gt(pdays_median).astype(int)
df['previous'] = df['previous'].gt(previous_median).astype(int)

binary = {'no': 0, 'yes': 1}
label_binary = {'no': -1, 'yes': 1}

df["default"] = df["default"].map(binary).astype(int)
df["loan"] = df["loan"].map(binary).astype(int)
df["housing"] = df["housing"].map(binary).astype(int)
df["y"] = df["y"].map(binary).astype(int)
sys.displayhook(df)