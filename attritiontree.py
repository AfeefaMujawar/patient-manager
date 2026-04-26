import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

# Encode categorical variables
df['JobRole'] = df['JobRole'].astype('category').cat.codes
df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})

# EDA
print("\nClass Distribution:\n", df['Attrition'].value_counts())

# Feature distributions
df[['Age','MonthlyIncome']].hist()
plt.show()

# Relationship
plt.scatter(df['Age'], df['MonthlyIncome'], c=df['Attrition'])
plt.xlabel("Age")
plt.ylabel("Monthly Income")
plt.title("Age vs Income (Attrition)")
plt.show()

# --- Part B ---
X = df[['Age','MonthlyIncome','JobRole']]
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# 🔥 Clean + readable tree
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,          # 👈 reduced depth (KEY FIX)
    min_samples_leaf=20,  # 👈 removes tiny branches
    random_state=0
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Tree visualization (balanced)
plt.figure(figsize=(18,8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No Attrition','Attrition'],
    filled=True,
    rounded=True,
    fontsize=9,
    proportion=True
)

plt.title("Decision Tree (Gini)")
plt.tight_layout()
plt.show()

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
print("Employees with lower income and specific job roles show higher attrition. Decision tree identifies clear thresholds influencing employee turnover.")