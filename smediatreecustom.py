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
np.random.seed(8)
n = 150

df = pd.DataFrame({
    'PostsPerDay': np.random.randint(1, 10, n),
    'Likes': np.random.randint(50, 1000, n),
    'Comments': np.random.randint(5, 200, n)
})

# Realistic engagement with noise
score = (
    (df['Likes'] > 400).astype(int) +
    (df['Comments'] > 50).astype(int) +
    (df['PostsPerDay'] >= 3).astype(int)
)

prob = 0.25 + 0.15 * score
rand = np.random.rand(n)

df['Engagement'] = np.where(rand < prob, 'High', 'Low')

# ===============================
# PREPROCESSING
# ===============================
print("\nMissing Values:\n", df.isnull().sum())

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

df[['PostsPerDay','Likes','Comments']].hist(figsize=(8,5))
plt.show()

df['Engagement'].value_counts().plot(kind='bar')
plt.title("Engagement Distribution")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================

# Encode target
df['Engagement'] = df['Engagement'].map({'Low':0,'High':1})

X = df[['PostsPerDay','Likes','Comments']]
y = df['Engagement']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=8, stratify=y
)

# Decision Tree (Entropy)
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    min_samples_split=10,
    random_state=8
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
    class_names=['Low','High'],
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
print("Higher likes and comments strongly indicate high engagement.")
print("Posting consistently also improves engagement levels.")
print("Decision Tree extracts clear engagement rules from user activity.")
print("Model shows realistic performance (not extreme 0 or 1).")