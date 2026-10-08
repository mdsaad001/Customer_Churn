import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("data/Churn_Modelling.csv")

print(df.head())
print("Shape:", df.shape)
print("Columns:", df.columns)

# Remove columns that are not useful for prediction
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Convert categorical columns into numeric dummy variables
df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True
)

print(df.head())

# Separate features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and train Logistic Regression model
model = LogisticRegression(
    max_iter=5000,
    solver="liblinear"
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
