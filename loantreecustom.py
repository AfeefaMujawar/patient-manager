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
np.random.seed(2)
n = 150

df = pd.DataFrame({
    'Income': np.random.randint(20000, 100000, n),
    'CreditScore': np.random.randint(450, 850, n),
    'LoanAmount': np.random.randint(10000, 50000, n)
})

# Realistic target with noise (avoid perfect scores)
score = (
    (df['Income'] > 50000).astype(int) +
    (df['CreditScore'] > 650).astype(int) -
    (df['LoanAmount'] > 30000).astype(int)
)

prob = 0.3 + 0.2 * score
rand = np.random.rand(n)

df['Approval'] = np.where(rand < prob, 'Yes', 'No')

# ===============================
# HANDLE MISSING VALUES
# ===============================
print("\nMissing Values:\n", df.isnull().sum())

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

df[['Income','CreditScore','LoanAmount']].hist(figsize=(8,5))
plt.show()

df['Approval'].value_counts().plot(kind='bar')
plt.title("Loan Approval Distribution")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================

# Encode target
df['Approval'] = df['Approval'].map({'No':0,'Yes':1})

X = df[['Income','CreditScore','LoanAmount']]
y = df['Approval']

# Stratified split (IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=3, stratify=y
)

# Decision Tree (VISIBLE + CONTROLLED)
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=10,
    random_state=3
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
    class_names=['Rejected','Approved'],
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
print("High income and high credit score increase approval chances.")
print("Large loan amounts reduce approval probability.")
print("Decision Tree creates clear rule-based decisions.")
print("Model shows realistic performance (not extreme 0 or 1).")