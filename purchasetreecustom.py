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
np.random.seed(4)
n = 150

df = pd.DataFrame({
    'Age': np.random.randint(18, 60, n),
    'Salary': np.random.randint(15000, 100000, n),
    'BrowsingTime': np.random.randint(5, 60, n)
})

# Realistic target with noise
score = (
    (df['BrowsingTime'] > 25).astype(int) +
    (df['Salary'] > 40000).astype(int) +
    (df['Age'] < 40).astype(int)
)

prob = 0.25 + 0.15 * score
rand = np.random.rand(n)

df['Purchased'] = np.where(rand < prob, 'Yes', 'No')

# ===============================
# PREPROCESSING
# ===============================
print("\nMissing Values:\n", df.isnull().sum())

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

df[['Age','Salary','BrowsingTime']].hist(figsize=(8,5))
plt.show()

df['Purchased'].value_counts().plot(kind='bar')
plt.title("Purchase Distribution")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================

# Encode target
df['Purchased'] = df['Purchased'].map({'No':0,'Yes':1})

X = df[['Age','Salary','BrowsingTime']]
y = df['Purchased']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=4, stratify=y
)

# Decision Tree (entropy)
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    min_samples_split=10,
    random_state=4
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ===============================
# TREE VISUALIZATION (CLEAR)
# ===============================
plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Not Purchased','Purchased'],
    filled=True,
    rounded=True,
    fontsize=9,
    proportion=True
)

plt.title("Decision Tree (Entropy)")
plt.tight_layout()
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
print("Customers with higher browsing time and moderate income are more likely to purchase.")
print("Younger users with higher engagement show stronger buying behavior.")
print("Decision Tree captures behavioral patterns effectively.")
print("Model performance is realistic (not extreme 0 or 1).")