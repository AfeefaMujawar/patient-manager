import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# --- Create Synthetic Dataset ---
data = [
[7.4,3.5,0.56,5],
[7.8,3.2,0.68,6],
[7.8,3.1,0.65,5],
[11.2,3.2,0.58,6],
[7.4,3.4,0.56,5],
[7.9,3.3,0.60,6],
[7.3,3.6,0.58,5],
[7.8,3.3,0.62,6],
[7.5,3.4,0.55,5],
[6.7,3.1,0.47,4]
]

df = pd.DataFrame(data, columns=['Alcohol','pH','Sulphates','Quality'])

# --- Part A ---
# Handle missing values
df = df.dropna()

# Statistical analysis
print("Statistical Summary:\n", df.describe())

# Visualization
df.hist()
plt.show()

# --- Part B ---
# Convert quality into categories
df['QualityLabel'] = df['Quality'].apply(lambda x: 'Low' if x <=5 else 'High')

# Encode labels
df['QualityLabel'] = df['QualityLabel'].astype('category').cat.codes

X = df[['Alcohol','pH','Sulphates']]
y = df['QualityLabel']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

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
print("Model predicts wine quality (Low/High) based on chemical features.")