import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("autos_dataset.csv")
df["num-of-cylinders"] = df["num-of-cylinders"].replace(
    {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "twelve": 12}
)
df.replace({"?": np.nan}, inplace=True)

# Same columns dropped as in the notebook
df.drop(["make", "fuel-type", "aspiration", "engine-type", "normalized-losses",
         "num-of-doors", "bore", "stroke", "fuel-system", "body-style",
         "drive-wheels", "engine-location"], axis=1, inplace=True)

for c in ["price", "peak-rpm", "horsepower"]:
    df[c] = df[c].astype(float)

df = df.dropna(subset=["price"])  # don't train on made-up target values

features = [c for c in df.columns if c != "price"]
X, y = df[features], df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", DecisionTreeRegressor(random_state=42)),
])

params = {
    "model__splitter": ["best", "random"],
    "model__max_depth": [1, 3, 5, 7, 9, 11, 12],
    "model__min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "model__max_features": ["log2", "sqrt", None],  # "auto" removed in new sklearn
    "model__max_leaf_nodes": [None, 10, 20, 30, 40, 50],
}

grid = GridSearchCV(pipe, params, scoring="neg_mean_squared_error", cv=3, n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)

pred = grid.best_estimator_.predict(X_test)
print("Test RMSE:", round(float(np.sqrt(mean_squared_error(y_test, pred))), 2))
print("Test MAE :", round(float(mean_absolute_error(y_test, pred)), 2))
print("Test R2  :", round(float(r2_score(y_test, pred)), 4))

with open("autos_model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)
with open("autos_features.pkl", "wb") as f:
    pickle.dump(features, f)
print("Features:", features)
print("Saved autos_model.pkl and autos_features.pkl")
