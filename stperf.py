import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# --- Synthetic Dataset (10 records) ---
data = {
    'StudyHours':[2,4,6,8,1,5,7,3,9,10],
    'Attendance':[60,70,80,90,50,75,85,65,95,98],
    'Marks':[40,55,65,80,35,60,75,50,85,90],
    'Result':['Fail','Fail','Pass','Pass','Fail','Pass','Pass','Fail','Pass','Pass']
}
df = pd.DataFrame(data)

# --- Part A ---
# Encode target
df['Result'] = df['Result'].map({'Fail':0,'Pass':1})

# EDA
print("\nDataset:\n", df)
print("\nClass Distribution:\n", df['Result'].value_counts())

# Distribution
df[['StudyHours','Attendance','Marks']].hist()
plt.show()

# Relationship
plt.scatter(df['StudyHours'], df['Marks'], c=df['Result'])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()

# --- Part B ---
X = df[['StudyHours','Attendance','Marks']]
y = df['Result']

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
    class_names=['Fail','Pass'],
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
print("Students with higher study hours and attendance tend to score higher marks and pass. Decision tree shows clear thresholds for passing.")