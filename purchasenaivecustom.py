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
    'Age':[18,22,25,30,35,40,28,45,50,27],
    'Salary':[15000,20000,25000,40000,50000,60000,30000,70000,80000,28000],
    'BrowsingTime':[5,10,15,20,25,30,18,35,40,12],
    'Purchased':['No','No','No','Yes','Yes','Yes','No','Yes','Yes','No']
}
df = pd.DataFrame(data)

# ===============================
# PREPROCESSING
# ===============================
print("\nMissing Values:\n", df.isnull().sum())

# Encode target
df['Purchased'] = df['Purchased'].map({'No':0,'Yes':1})

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

# Feature distributions
df[['Age','Salary','BrowsingTime']].hist()
plt.show()

# Class balance
df['Purchased'].value_counts().plot(kind='bar')
plt.title("Purchase Distribution")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================
X = df[['Age','Salary','BrowsingTime']]
y = df['Purchased']

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
print("Customers with higher browsing time and salary are more likely to purchase.")
print("Younger users with low engagement show lower purchase probability.")
print("Naive Bayes effectively predicts purchase behavior.")