# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ===============================
# PART A: SYNTHETIC DATASET (10)
# ===============================
data = {
    'Age':[25,30,45,50,35,40,60,55,28,48],
    'BMI':[22,25,30,32,27,29,35,34,23,31],
    'BloodPressure':[80,85,90,95,88,92,100,98,82,93],
    'Disease':['No','No','Yes','Yes','No','Yes','Yes','Yes','No','Yes']
}
df = pd.DataFrame(data)

# Handle missing values
print("\nMissing Values:\n", df.isnull().sum())

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

# Feature distributions
df[['Age','BMI','BloodPressure']].hist()
plt.show()

# Class distribution
df['Disease'].value_counts().plot(kind='bar')
plt.title("Disease Distribution")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================

# Encode target
df['Disease'] = df['Disease'].map({'No':0,'Yes':1})

X = df[['Age','BMI','BloodPressure']]
y = df['Disease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
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
print("Higher BMI and blood pressure increase disease risk.")
print("Older individuals are more likely to have disease.")
print("Model predicts disease based on health indicators effectively.")