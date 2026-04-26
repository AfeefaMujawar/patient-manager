import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# --- Create Synthetic Dataset ---
data = [
[5.1,3.5,1.4,0.2,'Setosa'],
[4.9,3.0,1.4,0.2,'Setosa'],
[5.8,2.7,5.1,1.9,'Virginica'],
[6.0,2.2,5.0,1.5,'Virginica'],
[5.5,2.3,4.0,1.3,'Versicolor'],
[6.5,2.8,4.6,1.5,'Versicolor'],
[5.7,2.8,4.5,1.3,'Versicolor'],
[6.3,3.3,6.0,2.5,'Virginica'],
[4.7,3.2,1.3,0.2,'Setosa'],
[5.0,3.6,1.4,0.2,'Setosa']
]

df = pd.DataFrame(data, columns=['SepalLength','SepalWidth','PetalLength','PetalWidth','Species'])

# --- Part A: EDA ---
print("Missing Values:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe())

# Histograms
df.hist()
plt.show()

# Class distribution
print("\nClass Distribution:\n", df['Species'].value_counts())

# --- Part B: Model Building ---
# Encode target
df['Species'] = df['Species'].astype('category').cat.codes

X = df[['SepalLength','SepalWidth','PetalLength','PetalWidth']]
y = df['Species']

# Split (70:30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# --- Part C: Evaluation ---
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro', zero_division=1))
print("Recall:", recall_score(y_test, y_pred, average='macro', zero_division=1))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Plot confusion matrix
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# Interpretation
print("\nInterpretation:")
print("Model performance shows how well species are classified based on flower features.")