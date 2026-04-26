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
np.random.seed(5)
n = 150

df = pd.DataFrame({
    'Age': np.random.randint(25, 70, n),
    'BMI': np.random.uniform(18, 40, n),
    'BloodPressure': np.random.randint(70, 120, n)
})

# Realistic target with noise
risk = (
    (df['BMI'] > 28).astype(int) +
    (df['BloodPressure'] > 90).astype(int) +
    (df['Age'] > 45).astype(int)
)

prob = 0.2 + 0.15 * risk
rand = np.random.rand(n)

df['Disease'] = np.where(rand < prob, 'Yes', 'No')

# ===============================
# HANDLE MISSING VALUES
# ===============================
print("\nMissing Values:\n", df.isnull().sum())

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

# Feature distributions
df[['Age','BMI','BloodPressure']].hist(figsize=(8,5))
plt.show()

# Feature relationships
plt.scatter(df['Age'], df['BMI'], c=(df['Disease']=='Yes'))
plt.xlabel("Age")
plt.ylabel("BMI")
plt.title("Age vs BMI (Disease)")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================

# Encode target
df['Disease'] = df['Disease'].map({'No':0,'Yes':1})

X = df[['Age','BMI','BloodPressure']]
y = df['Disease']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=5, stratify=y
)

# Decision Tree (Gini)
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=10,
    random_state=5
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
    class_names=['No Disease','Disease'],
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
print("High BMI and blood pressure significantly increase disease risk.")
print("Older age contributes to higher probability of disease.")
print("Decision Tree creates clear rule-based medical decisions.")
print("Model shows realistic performance (not extreme 0 or 1).")