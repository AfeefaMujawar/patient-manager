import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv", encoding='latin1')

# Clean column names (important)
df.columns = df.columns.str.strip().str.replace(' ', '_')

# --- Part A ---
# Select required columns
df = df[['ApplicantIncome','LoanAmount','Gender','Loan_Status']]

# Handle missing values
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

# Encode categorical
df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
df['Loan_Status'] = df['Loan_Status'].map({'Y':1,'N':0})

# EDA
print("\nClass Distribution:\n", df['Loan_Status'].value_counts())

df[['ApplicantIncome','LoanAmount']].hist()
plt.show()

pd.crosstab(df['Gender'], df['Loan_Status']).plot(kind='bar')
plt.title("Gender vs Loan Status")
plt.show()

# --- Part B ---
X = df[['ApplicantIncome','LoanAmount','Gender']]
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# Decision Tree (gini)
model = DecisionTreeClassifier(
    criterion='gini',
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
    class_names=['Rejected','Approved'],
    filled=True,
    fontsize=10
)
plt.title("Decision Tree (Gini)")
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
print("Decision Tree predicts loan approval using income, loan amount, and gender. Higher income improves approval chances.")