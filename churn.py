import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# --- Create Synthetic Dataset ---
data = [
[600,'France','Male',40,60000,0],
[700,'Germany','Female',42,80000,1],
[650,'Spain','Male',35,50000,0],
[720,'France','Female',50,90000,1],
[580,'Germany','Male',30,30000,0],
[690,'Spain','Female',45,70000,1],
[710,'France','Male',48,85000,1],
[640,'Germany','Female',38,62000,0],
[680,'Spain','Male',41,72000,1],
[660,'France','Female',36,58000,0]
]

df = pd.DataFrame(data, columns=[
    'CreditScore','Geography','Gender','Age','Balance','Exited'
])

# --- Part A ---
# Cleaning
df = df.dropna()

# Encode categorical
df['Geography'] = df['Geography'].astype('category').cat.codes
df['Gender'] = df['Gender'].map({'Male':0,'Female':1})

# Class distribution
print("\nClass Distribution:\n", df['Exited'].value_counts())

# Visualization
df[['CreditScore','Age','Balance']].hist()
plt.show()

pd.crosstab(df['Gender'], df['Exited']).plot(kind='bar')
plt.title("Gender vs Churn")
plt.show()

# --- Part B ---
X = df[['CreditScore','Geography','Gender','Age','Balance']]
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0, stratify=y
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
plt.figure(figsize=(18,8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No Churn','Churn'],
    filled=True,
    fontsize=10
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
print("Model predicts customer churn based on credit score, age, balance, and demographics. High balance and age often indicate higher churn risk.")