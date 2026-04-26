import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Social_Network_Ads.csv")
df.columns = df.columns.str.strip()

# --- Part A ---
# Select required columns
df = df[['Age','EstimatedSalary','Purchased']]

# Clean dataset
df = df.dropna()

# Distribution
print("\nClass Distribution:\n", df['Purchased'].value_counts())

# Feature distributions
df[['Age','EstimatedSalary']].hist()
plt.show()

# Relationship with target
plt.scatter(df['Age'], df['EstimatedSalary'], c=df['Purchased'])
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Age vs Salary (Colored by Purchase)")
plt.show()

# --- Part B ---
X = df[['Age','EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# Decision Tree (gini)
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    random_state=0
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Tree visualization
plt.figure(figsize=(18,8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No','Yes'],
    filled=True,
    fontsize=10
)
plt.title("Decision Tree (Gini)")
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
print("Decision Tree predicts purchase behavior based on age and salary. Higher salary and middle-age users are more likely to purchase.")