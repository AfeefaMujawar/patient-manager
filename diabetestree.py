import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("diabetes.csv")
df.columns = df.columns.str.strip()

# --- Part A ---
# Select required columns
df = df[['Glucose','BMI','Age','Outcome']]

# Handle missing values (0 → NaN → median)
for col in ['Glucose','BMI','Age']:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# EDA
print("\nSummary:\n", df.describe())
print("\nClass Distribution:\n", df['Outcome'].value_counts())

# Visualization
df[['Glucose','BMI','Age']].hist()
plt.show()

# Relationship
plt.scatter(df['Glucose'], df['BMI'])
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("Glucose vs BMI")
plt.show()

# --- Part B ---
X = df[['Glucose','BMI','Age']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# Decision Tree (entropy)
model = DecisionTreeClassifier(
    criterion='entropy',
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
    class_names=['No Diabetes','Diabetes'],
    filled=True,
    fontsize=10
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
print("Decision Tree predicts diabetes based on glucose, BMI, and age. Higher glucose levels strongly indicate diabetes risk.")