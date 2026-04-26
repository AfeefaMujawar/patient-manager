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
np.random.seed(6)
n = 150

df = pd.DataFrame({
    'CGPA': np.round(np.random.uniform(5.0, 9.5, n), 2),
    'Internships': np.random.randint(0, 4, n),
    'SkillsScore': np.random.randint(50, 100, n)
})

# Realistic target with noise
score = (
    (df['CGPA'] > 7).astype(int) +
    (df['Internships'] >= 1).astype(int) +
    (df['SkillsScore'] > 70).astype(int)
)

prob = 0.25 + 0.15 * score
rand = np.random.rand(n)

df['Placed'] = np.where(rand < prob, 'Yes', 'No')

# ===============================
# PREPROCESSING
# ===============================
print("\nMissing Values:\n", df.isnull().sum())

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

df[['CGPA','Internships','SkillsScore']].hist(figsize=(8,5))
plt.show()

df['Placed'].value_counts().plot(kind='bar')
plt.title("Placement Distribution")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================

# Encode target
df['Placed'] = df['Placed'].map({'No':0,'Yes':1})

X = df[['CGPA','Internships','SkillsScore']]
y = df['Placed']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=6, stratify=y
)

# Decision Tree (Entropy)
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    min_samples_split=10,
    random_state=6
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
    class_names=['Not Placed','Placed'],
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
print("Higher CGPA, internships, and skills increase placement probability.")
print("Students with low skills and no internships are less likely to be placed.")
print("Decision Tree forms clear rule-based placement decisions.")
print("Model shows realistic performance (not extreme 0 or 1).")