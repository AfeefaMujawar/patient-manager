import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("heart.csv")
df.columns = df.columns.str.strip()

# --- Part A ---
# Select required columns
df = df[['age','chol','trestbps','target']]

# Handle missing values (DO NOT replace all 0s)
df = df.fillna(df.median(numeric_only=True))

# EDA
print("\nSummary:\n", df.describe())
print("\nClass Distribution:\n", df['target'].value_counts())

# Feature distributions
df[['age','chol','trestbps']].hist()
plt.show()

# Feature relationships
plt.scatter(df['age'], df['chol'])
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.title("Age vs Cholesterol")
plt.show()

# --- Part B ---
X = df[['age','chol','trestbps']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# 🔥 Improved Decision Tree
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    min_samples_split=10,
    random_state=0
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Tree visualization (clean + readable)
plt.figure(figsize=(20,10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No Disease','Disease'],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree (Entropy)")
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
print("Decision Tree classifies heart disease using age, cholesterol, and blood pressure. Higher cholesterol and older age increase disease risk.")