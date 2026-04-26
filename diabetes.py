import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("diabetes.csv")
df.columns = df.columns.str.strip()

# --- Part A ---
# Handle zero values (replace with median)
cols = ['Glucose','BloodPressure','BMI','Age']
for col in cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# EDA
print("Statistical Summary:\n", df.describe())

# Visualization
df[cols].hist()
plt.show()

# Class imbalance
print("\nClass Distribution:\n", df['Outcome'].value_counts())

# --- Part B ---
X = df[['Glucose','BloodPressure','BMI','Age']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# --- Part C ---
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=1))
print("Recall:", recall_score(y_test, y_pred, zero_division=1))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# Interpretation
print("\nInterpretation:")
print("Model predicts diabetes based on medical features. Performance depends on feature quality and class balance.")