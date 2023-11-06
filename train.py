
# Import Libraries
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 800)
pd.set_option('display.max_columns', 500)


df = pd.read_csv("./MonkeyPox.csv")

df.columns = df.columns.str.lower().str.replace(' ', '_')


# let's drop the patien_id column which won't contribute to the model using domain knowldedge.
df.drop(['patient_id'], axis=1, inplace=True)

cat_cols = list(df.dtypes[df.dtypes == 'object'].index)
print(f'Columns with categorical variables are: {cat_cols} \n')


bool_cols = list(df.dtypes[df.dtypes == 'bool'].index)
print(f'Columns with boolean are: {bool_cols}')


# convert boolean to integers
df[bool_cols] = df[bool_cols].astype(int)

df['monkeypox'] = df['monkeypox'].replace(['Positive', 'Negative'], [1, 0])

# split data set to 60/20/20 for validation and testing
data_full_train, data_test = train_test_split(df, test_size=0.2, random_state=42)
data_train, data_val = train_test_split(data_full_train, test_size=0.25, random_state=42)

len(data_train), len(data_val), len(data_test)

y_train = data_train.monkeypox.values
y_val = data_val.monkeypox.values
y_test = data_test.monkeypox.values

del data_train['monkeypox']
del data_val['monkeypox']
del data_test['monkeypox']


cat_col = ['systemic_illness']


dicts_full_train = data_full_train[bool_cols + cat_col].to_dict(orient='records')


dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train).astype(int)
y_full_train = data_full_train.monkeypox.values

print(y_full_train)
modelrf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1, max_depth=5)
modelrf.fit(X_full_train, y_full_train)


dicts_test = data_test[bool_cols + cat_col].to_dict(orient='records')
X_test = dv.transform(dicts_test).astype(int)

rfp_pred = modelrf.predict(X_test)
rfp_class = classification_report(y_test, rfp_pred)
rfp_score = accuracy_score(y_test, rfp_pred)
print(rfp_class)
print(f'\n Improved accuracy score of {rfp_score} with RandomForest Classifier')


# Save the model

with open('model.pkl', 'wb') as f_out:
    pickle.dump((dv, modelrf), f_out)

print(f'the model is saved to model.pkl')
