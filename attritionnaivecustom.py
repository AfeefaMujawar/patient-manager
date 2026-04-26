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
    'Age':[22,25,30,35,40,28,32,45,50,29],
    'Salary':[20000,25000,40000,50000,60000,35000,45000,70000,80000,30000],
    'Experience':[1,2,5,8,10,3,6,12,15,4],
    'Attrition':['Yes','Yes','No','No','No','Yes','No','No','No','Yes']
}
df = pd.DataFrame(data)

# ===============================
# DATA CLEANING
# ===============================
print("\nMissing Values:\n", df.isnull().sum())

# Encode target
df['Attrition'] = df['Attrition'].map({'No':0,'Yes':1})

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

# Feature distributions
df[['Age','Salary','Experience']].hist()
plt.show()

# Class balance
df['Attrition'].value_counts().plot(kind='bar')
plt.title("Attrition Distribution")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================
X = df[['Age','Salary','Experience']]
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

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
print("Employees with lower salary and experience are more likely to leave.")
print("Higher salary and experienced employees show lower attrition.")
print("Naive Bayes effectively predicts attrition based on employee features.")