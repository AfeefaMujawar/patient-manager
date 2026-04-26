import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("bank-full.csv", sep=',')   # 🔥 IMPORTANT
df.columns = df.columns.str.strip()          # remove spaces
df.columns = df.columns.str.lower()          # normalize names

print(df.columns)  # debug

# --- Part A ---
# Select required columns
df = df[['age','job','balance','loan','y']]

# Handle missing values
df = df.dropna()

# Encode categorical features
df['job'] = df['job'].astype('category').cat.codes
df['loan'] = df['loan'].map({'yes':1, 'no':0})
df['y'] = df['y'].map({'yes':1, 'no':0})

# EDA
print("\nClass Distribution:\n", df['y'].value_counts())

# Feature distributions
df[['age','balance']].hist()
plt.show()

# Relationship
plt.scatter(df['age'], df['balance'], c=df['y'])
plt.xlabel("Age")
plt.ylabel("Balance")
plt.title("Age vs Balance (Subscription)")
plt.show()

# --- Part B ---
X = df[['age','job','balance','loan']]
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# Decision Tree (gini)
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,          # 🔥 clean tree
    min_samples_leaf=20,  # 🔥 removes noise
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
print("Decision Tree predicts customer subscription based on age, job, balance, and loan status. Customers with higher balance and no loans are more likely to subscribe.")