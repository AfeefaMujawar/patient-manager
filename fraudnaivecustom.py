# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ===============================
# PART A: SYNTHETIC DATASET (10)
# ===============================
data = {
    'TransactionAmount':[100,200,5000,7000,150,300,8000,120,4500,250],
    'Frequency':[2,3,1,1,5,4,1,6,2,5],
    'LocationScore':[0.2,0.3,0.9,0.95,0.25,0.4,0.98,0.2,0.85,0.35],
    'Fraud':['No','No','Yes','Yes','No','No','Yes','No','Yes','No']
}
df = pd.DataFrame(data)

# ===============================
# PREPROCESSING
# ===============================
print("\nMissing Values:\n", df.isnull().sum())

# Encode target
df['Fraud'] = df['Fraud'].map({'No':0,'Yes':1})

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

# Feature distributions
df[['TransactionAmount','Frequency','LocationScore']].hist()
plt.show()

# Class imbalance
df['Fraud'].value_counts().plot(kind='bar')
plt.title("Fraud Class Distribution")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================
X = df[['TransactionAmount','Frequency','LocationScore']]
y = df['Fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=3, stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

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
print("High transaction amount + unusual location score indicate fraud.")
print("Low frequency with high amount is suspicious behavior.")
print("Model provides balanced fraud detection without extreme results.")