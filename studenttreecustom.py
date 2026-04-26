# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ===============================
# PART A: SYNTHETIC DATASET (150)
# ===============================
np.random.seed(0)

n = 150
df = pd.DataFrame({
    'StudyHours': np.random.randint(1, 10, n),
    'Attendance': np.random.randint(50, 100, n),
    'AssignmentScore': np.random.randint(40, 100, n)
})

# Create realistic target
df['Result'] = np.where(
    (df['StudyHours'] > 5) & (df['Attendance'] > 70) & (df['AssignmentScore'] > 60),
    'Pass', 'Fail'
)

# ===============================
# HANDLE MISSING VALUES
# ===============================
df = df.fillna(df.mean(numeric_only=True))

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

# Feature distributions
df[['StudyHours','Attendance','AssignmentScore']].hist(figsize=(8,5))
plt.show()

# Target relationship
pd.crosstab(df['StudyHours'], df['Result']).plot(kind='bar', stacked=True)
plt.title("StudyHours vs Result")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================

# Encode target
df['Result'] = df['Result'].map({'Fail':0,'Pass':1})

X = df[['StudyHours','Attendance','AssignmentScore']]
y = df['Result']

# Split (70:30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# Decision Tree (VISIBLE TREE)
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,          # 🔥 makes tree readable
    min_samples_split=5,
    random_state=0
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ===============================
# TREE VISUALIZATION (CLEAR)
# ===============================
plt.figure(figsize=(18,8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Fail','Pass'],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree (Gini)")
plt.show()

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
print("Students pass when study hours, attendance, and assignment scores are high.")
print("Decision tree splits mainly on AssignmentScore and Attendance.")
print("Model provides clear decision rules for academic performance.")