# The data file CDCRDA (1).csv 


# Create a machine learning prediction model that predicts the final grade of a student based on the other attributes.
# Use the following steps to create your model:
# 1. Load the data from the file CDCRDA (1).csv
# 1.5. LabelEncoding the data
# 2. Split the data into a training and test set
# 3. Create a decision tree classifier and a random forest classifier
# 4. Train both models
# 5. Evaluate both models
# 6. Save the best model to a file called model.pkl
# make it as accurate as possible

# Predict if the student has depression or not based on the other attributes.

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd

# 1. Load the data from the file CDCRDA (1).csv
df = pd.read_csv("CDCRDA (1).csv")

# drop nan values
df = df.dropna()

# 1.5. LabelEncoding the data
le = LabelEncoder()
# Iterate over all the values of each column and extract their dtypes
for col in df.columns:
    # Compare if the dtype is object
    if df[col].dtype=='object':
    # Use LabelEncoder to do the numeric transformation
        df[col]=le.fit_transform(df[col])

# 2. Split the data into a training and test set
X = df.drop('depressed', axis=1)
y = df['depressed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Create a decision tree classifier and a random forest classifier
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()

# 4. Train both models
dtc.fit(X_train, y_train)
rfc.fit(X_train, y_train)

# 5. Evaluate both models
print(dtc.score(X_test, y_test))
print(rfc.score(X_test, y_test))

# 6. Save the best model to a file called model.pkl
if dtc.score(X_test, y_test) > rfc.score(X_test, y_test):
    pickle.dump(dtc, open('model.pkl','wb'))
else:
    pickle.dump(rfc, open('model.pkl','wb'))

