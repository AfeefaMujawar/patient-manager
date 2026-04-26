import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("titanictrain.csv", encoding='latin1')

# Clean column names (important fix)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '')

# --- Part A ---
# Select required columns
df = df[['age','sex','fare','pclass','survived']]

# Handle missing values
df['age'] = df['age'].fillna(df['age'].median())

# Encode categorical
df['sex'] = df['sex'].map({'male':0, 'female':1})

# Distribution
print("\nSurvival Distribution:\n", df['survived'].value_counts())

# Feature distributions
df[['age','fare']].hist()
plt.show()

# Relationship (Sex vs Survived)
pd.crosstab(df['sex'], df['survived']).plot(kind='bar')
plt.title("Sex vs Survival")
plt.show()

# --- Part B ---
X = df[['age','sex','fare','pclass']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# 🔥 Improved readable tree
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    random_state=0
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Visualize tree
plt.figure(figsize=(20,10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Dead','Survived'],
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
print("Decision Tree predicts survival using age, gender, fare, and class. Gender and passenger class strongly influence survival outcomes.")