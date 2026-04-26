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
np.random.seed(9)
n = 150

df = pd.DataFrame({
    'Temperature': np.random.randint(50, 120, n),
    'Pressure': np.random.randint(20, 100, n),
    'DefectRate': np.round(np.random.uniform(0.01, 0.2, n), 3)
})

# Realistic quality logic with noise
risk = (
    (df['Temperature'] > 90).astype(int) +
    (df['Pressure'] > 70).astype(int) +
    (df['DefectRate'] > 0.1).astype(int)
)

prob = 0.25 + 0.15 * risk
rand = np.random.rand(n)

df['Quality'] = np.where(rand < prob, 'Bad', 'Good')

# ===============================
# PREPROCESSING
# ===============================
print("\nMissing Values:\n", df.isnull().sum())

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

# Distributions
df[['Temperature','Pressure','DefectRate']].hist(figsize=(8,5))
plt.show()

# Feature relationships
plt.scatter(df['Temperature'], df['DefectRate'], c=(df['Quality']=='Bad'))
plt.xlabel("Temperature")
plt.ylabel("Defect Rate")
plt.title("Temperature vs DefectRate (Quality)")
plt.show()

# Class balance
df['Quality'].value_counts().plot(kind='bar')
plt.title("Quality Distribution")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================

# Encode target
df['Quality'] = df['Quality'].map({'Good':0,'Bad':1})

X = df[['Temperature','Pressure','DefectRate']]
y = df['Quality']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=9, stratify=y
)

# Decision Tree (Gini)
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=10,
    random_state=9
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
    class_names=['Good','Bad'],
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
print("High temperature, high pressure, and high defect rate lead to poor quality.")
print("Stable conditions produce good quality output.")
print("Decision Tree extracts clear production rules.")
print("Model performance is realistic (not extreme 0 or 1).")