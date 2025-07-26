import os
import sys
import tarfile
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import joblib

# Define the absolute path for the web_app directory
WEB_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web_app'))
sys.path.append(WEB_APP_DIR)

# Now that the path is added, import the custom transformer
from custom_transformer import CombinedAttributesAdder

# Define paths
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    # Check if the tgz file exists, if not, download it
    if not os.path.exists(tgz_path):
        print("Downloading housing data...")
        urllib.request.urlretrieve(housing_url, tgz_path)
    # Check if the csv file exists, if not, extract it
    if not os.path.exists(os.path.join(housing_path, "housing.csv")):
        print("Extracting housing data...")
        with tarfile.open(tgz_path) as housing_tgz:
            housing_tgz.extractall(path=housing_path)
        print("Data extracted.")
    else:
        print("Data already exists.")

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# --- Main execution ---
print("Starting model recreation process...")

# 1. Fetch and load data
fetch_housing_data()
housing = load_housing_data()

# 2. Create income categories for stratified split
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# 3. Perform stratified split
print("Performing stratified split...")
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

housing_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

# 4. Create the full data processing pipeline
print("Creating data processing pipeline...")
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# 5. Fit the pipeline and transform the data
print("Fitting pipeline...")
full_pipeline.fit(housing)

# 6. Define model directory and save the pipeline
MODEL_DIR = os.path.join(WEB_APP_DIR, "model")
if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

PIPELINE_PATH = os.path.join(MODEL_DIR, "full_pipeline.joblib")
print(f"Saving pipeline to {PIPELINE_PATH}")
joblib.dump(full_pipeline, PIPELINE_PATH)

# 7. Prepare the data for training
housing_prepared = full_pipeline.transform(housing)

# 8. Train the RandomForestRegressor model
print("Training RandomForestRegressor model...")
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
forest_reg.fit(housing_prepared, housing_labels)

# 9. Save the trained model
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
print(f"Saving model to {MODEL_PATH}")
joblib.dump(forest_reg, MODEL_PATH)

print("\nModel and pipeline have been recreated and saved successfully in web_app/model/")
