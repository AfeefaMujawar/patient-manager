import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# --- Synthetic Dataset (10 records) ---
data = {
    'Radius':[12,14,10,18,20,9,16,11,22,13],
    'Texture':[15,18,12,22,25,10,20,14,27,16],
    'Perimeter':[80,90,70,110,130,65,105,75,140,85],
    'Diagnosis':['Benign','Benign','Benign','Malignant','Malignant',
                 'Benign','Malignant','Benign','Malignant','Benign']
}
df = pd.DataFrame(data)

# --- Part A ---
# Handle missing values
df = df.fillna(df.median(numeric_only=True))

# Encode target
df['Diagnosis'] = df['Diagnosis'].map({'Benign':0,'Malignant':1})

# Statistical analysis
print("\nSummary:\n", df.describe())
print("\nClass Distribution:\n", df['Diagnosis'].value_counts())

# Feature distributions
df[['Radius','Texture','Perimeter']].hist()
plt.show()

# Relationship
plt.scatter(df['Radius'], df['Perimeter'], c=df['Diagnosis'])
plt.xlabel("Radius")
plt.ylabel("Perimeter")
plt.title("Radius vs Perimeter")
plt.show()

# --- Part B ---
X = df[['Radius','Texture','Perimeter']]
y = df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
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
plt.figure(figsize=(16,6))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Benign','Malignant'],
    filled=True,
    rounded=True,
    fontsize=9
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
print("Higher radius, texture, and perimeter values indicate malignant tumors. Decision tree identifies thresholds for diagnosis classification.")