# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ===============================
# PART A: SYNTHETIC DATASET (10)
# ===============================
data = {
    'Income':[20000,25000,40000,50000,60000,30000,45000,70000,80000,35000],
    'CreditScore':[500,550,650,700,750,600,680,780,820,620],
    'LoanAmount':[10000,12000,20000,25000,30000,15000,22000,35000,40000,18000],
    'Approval':['No','No','Yes','Yes','Yes','No','Yes','Yes','Yes','No']
}
df = pd.DataFrame(data)

# ===============================
# HANDLE MISSING VALUES
# ===============================
print("\nMissing Values:\n", df.isnull().sum())

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

# Feature distributions
df[['Income','CreditScore','LoanAmount']].hist()
plt.show()

# Class distribution
df['Approval'].value_counts().plot(kind='bar')
plt.title("Loan Approval Distribution")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================

# Encode target
df['Approval'] = df['Approval'].map({'No':0,'Yes':1})

X = df[['Income','CreditScore','LoanAmount']]
y = df['Approval']

# Train-test split (70:30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ===============================
# PART C: EVALUATION
# ===============================
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

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Applicants with higher income and credit score are more likely to be approved.")
print("Low credit score and low income increase rejection probability.")
print("Naive Bayes effectively predicts loan approval based on financial features.")