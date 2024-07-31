import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

data = pd.read_csv('Datasets/StudentScore.csv')

target = 'MathScore'
x = data.drop(data[[target, 'Unnamed: 0']], axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# numerical processing
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

x_train[['ReadingScore', 'WritingScore']] = num_transformer.fit_transform(x_train[['ReadingScore', 'WritingScore']])
x_test[['ReadingScore', 'WritingScore']] = num_transformer.transform(x_test[['ReadingScore', 'WritingScore']])

# ordinal processing
education_levels = ["high school", "some high school", "some college", "associate's degree", "bachelor's degree",
                    "master's degree"]

ord_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[education_levels])),
])

x_train[['ParentEduc']] = ord_transformer.fit_transform(x_train[['ParentEduc']])
x_test[['ParentEduc']] = ord_transformer.transform(x_test[['ParentEduc']])

# nominal processing
nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])

x_train[['EthnicGroup']] = nom_transformer.fit_transform(x_train[['EthnicGroup']])
x_test[['EthnicGroup']] = nom_transformer.transform(x_test[['EthnicGroup']])

# boolean processing
bool_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False)),
])

x_train[['Gender', 'LunchType', 'TestPrep']] = bool_transformer.fit_transform(
    x_train[['Gender', 'LunchType', 'TestPrep']])
x_test[['Gender', 'LunchType', 'TestPrep']] = bool_transformer.transform(x_train[['Gender', 'LunchType', 'TestPrep']])
