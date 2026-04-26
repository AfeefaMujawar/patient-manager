import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.columns = df.columns.str.strip()

# --- Part A ---
# Select required columns
df = df[['Age','MonthlyIncome','JobRole','Attrition']]

# Handle missing values
df = df.dropna()

# Encode categorical
df['JobRole'] = df['JobRole'].astype('category').cat.codes
df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})

# EDA
print("\nClass Distribution:\n", df['Attrition'].value_counts())

df[['Age','MonthlyIncome']].hist()
plt.show()

# --- Part B ---
X = df[['Age','MonthlyIncome','JobRole']]
y = df['Attrition']

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
print("Model predicts employee attrition based on age, income, and job role patterns.")