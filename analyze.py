# import the necessary packages
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time

# load the model.pkl file
model = pickle.load(open('model.pkl', 'rb'))

# open a dataframe from mydata.csv with the same columns as the training data
df = pd.read_csv("mydata.csv")

# encode the dataframe
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype=='object':
        df[col]=le.fit_transform(df[col])

# predict the result
result = model.predict(df)

# print the result
if result[0] == 1:
    print("You have depression")
else:
    print("You don't have depression")

# print the probability of the result
print(model.predict_proba(df))
