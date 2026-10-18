import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import joblib

print("Starting model generation...")

# 1. Load the actual Titanic training dataset
df = sns.load_dataset('titanic')[["survived", "pclass", "sex", "age", "fare"]].dropna()

# 2. Convert text columns to numerical values matching the app inputs
df["sex"] = (df["sex"] == "female").astype(int)

X = df[["pclass", "sex", "age", "fare"]]
y = df["survived"]

# 3. Train the machine learning classifier model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)

# 4. Save out the binary file so app.py can load it instantly
joblib.dump(model, 'model.joblib')
print("Success! 'model.joblib' has been successfully compiled and saved.")
