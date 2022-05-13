import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import pickle


def shorten_categories(categories, cutoff):
    new_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            new_map[categories.index[i]] = categories.index[i]
        else:
            new_map[categories.index[i]] = "Other"

    return new_map


def year_to_int(x):
    if x == 'More than 50 years':
        return 0.5

    if x == 'Less than 1 year':
        return 1

    return float(x)


def reduce_education(x):
    if "Bachelor’s" in x:
        return "Bachelor's degree"

    if "Master" in x:
        return "Master's degree"

    if "Professional" in x or "Other doctoral" in x:
        return "Post grad"

    return "Less than a Bachelors"


pd.set_option('display.max_columns', 10)
df = pd.read_csv('survey_results_public.csv')

df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "CompTotal"]]

df = df[df["CompTotal"].notnull()]

df.dropna(inplace=True)

df = df[df["Employment"] == "Employed full-time"]
df.drop("Employment", axis=1, inplace=True)

reduce_map = shorten_categories(df["Country"].value_counts(), 400)
df["Country"] = df["Country"].map(reduce_map)
df = df[df["Country"] != "Other"]

df = df[df["CompTotal"] <= 500000]

df["YearsCodePro"] = df["YearsCodePro"].apply(year_to_int)
df["EdLevel"] = df["EdLevel"].apply(reduce_education)

saved_df = {"df": df}
with open('saved_df.pkl', 'wb') as file:
    pickle.dump(df, file)


le_education = LabelEncoder()
df["EdLevel"] = le_education.fit_transform(df["EdLevel"])

le_country = LabelEncoder()

df["Country"] = le_country.fit_transform(df["Country"])

X = np.array(df.drop("CompTotal", axis=1).values)
y = np.array(df["CompTotal"].values)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

max_depth = [None, 2, 4, 6, 8, 10, 12]
parameters = {"max_depth": max_depth}

model = DecisionTreeRegressor(random_state=0)
gs = GridSearchCV(model, parameters, scoring="neg_mean_squared_error")
gs.fit(X, y)

regressor = gs.best_estimator_

regressor.fit(X, y)

data = {"model": regressor, "le_country": le_country, "le_education": le_education}

with open('saved_model.pkl', 'wb') as file:
    pickle.dump(data, file)
