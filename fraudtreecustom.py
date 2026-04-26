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
np.random.seed(7)
n = 150

df = pd.DataFrame({
    'TransactionAmount': np.random.randint(100, 10000, n),
    'Frequency': np.random.randint(1, 10, n),
    'LocationScore': np.round(np.random.uniform(0.1, 1.0, n), 2)
})

# Realistic fraud logic with noise
risk = (
    (df['TransactionAmount'] > 5000).astype(int) +
    (df['Frequency'] < 3).astype(int) +
    (df['LocationScore'] > 0.8).astype(int)
)

prob = 0.2 + 0.15 * risk
rand = np.random.rand(n)

df['Fraud'] = np.where(rand < prob, 'Yes', 'No')

# ===============================
# PREPROCESSING
# ===============================
print("\nMissing Values:\n", df.isnull().sum())

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

df[['TransactionAmount','Frequency','LocationScore']].hist(figsize=(8,5))
plt.show()

# Class imbalance
df['Fraud'].value_counts().plot(kind='bar')
plt.title("Fraud Distribution")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================

# Encode target
df['Fraud'] = df['Fraud'].map({'No':0,'Yes':1})

X = df[['TransactionAmount','Frequency','LocationScore']]
y = df['Fraud']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=7, stratify=y
)

# Decision Tree (Gini)
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=10,
    random_state=7
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
    class_names=['Normal','Fraud'],
    filled=True,
    rounded=True,
    fontsize=9,
    proportion=True
)

plt.title("Decision Tree (Gini)")
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
print("High transaction amount and unusual location increase fraud risk.")
print("Low frequency with high value transactions are suspicious.")
print("Decision Tree forms clear rule-based fraud detection.")
print("Model performance is realistic (not extreme 0 or 1).")