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
    'CGPA':[6.5,7.0,8.0,8.5,9.0,6.8,7.5,8.2,9.1,7.2],
    'Internships':[0,1,2,2,3,1,2,3,3,1],
    'SkillsScore':[60,65,75,80,90,70,78,85,92,68],
    'Placed':['No','No','Yes','Yes','Yes','No','Yes','Yes','Yes','No']
}
df = pd.DataFrame(data)

# ===============================
# PREPROCESSING
# ===============================
print("\nMissing Values:\n", df.isnull().sum())

# Encode target
df['Placed'] = df['Placed'].map({'No':0,'Yes':1})

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

# Feature distributions
df[['CGPA','Internships','SkillsScore']].hist()
plt.show()

# Class distribution
df['Placed'].value_counts().plot(kind='bar')
plt.title("Placement Distribution")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================
X = df[['CGPA','Internships','SkillsScore']]
y = df['Placed']

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
print("Higher CGPA, more internships, and strong skills increase placement chances.")
print("Students with low CGPA and fewer internships are less likely to be placed.")
print("Model effectively predicts placement based on academic and skill factors.")