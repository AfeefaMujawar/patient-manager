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
# PART A: SYNTHETIC DATASET
# ===============================
data = {
    'StudyHours':[2,4,6,8,1,5,7,3,9,10],
    'Attendance':[60,70,80,90,50,75,85,65,95,98],
    'AssignmentScore':[50,60,70,85,40,65,78,55,90,92],
    'Result':['Fail','Fail','Pass','Pass','Fail','Pass','Pass','Fail','Pass','Pass']
}
df = pd.DataFrame(data)

# Handle missing values
print("\nMissing Values:\n", df.isnull().sum())

# Statistical summary
print("\nSummary:\n", df.describe())

# ===============================
# VISUALIZATION
# ===============================
df[['StudyHours','Attendance','AssignmentScore']].hist()
plt.show()

# Class distribution
df['Result'].value_counts().plot(kind='bar')
plt.title("Class Distribution")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================

# Encode target
df['Result'] = df['Result'].map({'Fail':0,'Pass':1})

X = df[['StudyHours','Attendance','AssignmentScore']]
y = df['Result']

# Train-test split (70:30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
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
print("Students with higher study hours, attendance, and assignment scores are more likely to pass.")
print("Naive Bayes effectively classifies students based on academic performance.")